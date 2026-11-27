"""
Literature ingestion: query -> collectors -> deduplicate -> research database (-> knowledge base).

Every search is logged as a SearchRun (status ok / cached / rate_limited / failed / skipped), so a
collection's coverage can be reported exactly. Publications without an abstract are kept and flagged
as metadata-only; full text is only fetched for open-access arXiv PDFs, one paper at a time, on request.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import func, or_, select

from ..document_processing.chunker import DocumentChunk, TraceableChunker
from ..storage import Publication, SearchHit, SearchRun, session_scope
from .collectors import COLLECTORS, BaseCollector, PublicationRecord
from .collectors.base import ROOT, SearchResult
from .deduplication import add_versions, find_existing, merge_into, new_publication

logger = logging.getLogger(__name__)
DEFAULT_SOURCES = ("openalex", "arxiv", "crossref")
FULLTEXT_DIR = os.path.join(ROOT, "data", "fulltext")


@dataclass
class SourceOutcome:
    source: str
    status: str
    returned: int = 0
    total_available: Optional[int] = None
    new: int = 0
    merged: int = 0
    error: Optional[str] = None
    search_run_id: Optional[int] = None


@dataclass
class IngestReport:
    query: str
    topic: Optional[str]
    filters: Dict[str, Any]
    outcomes: List[SourceOutcome] = field(default_factory=list)
    publication_ids: List[int] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None

    @property
    def new(self) -> int:
        return sum(o.new for o in self.outcomes)

    @property
    def merged(self) -> int:
        return sum(o.merged for o in self.outcomes)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(new=self.new, merged=self.merged, unique=len(set(self.publication_ids)))
        return d


class LiteratureIngestor:
    def __init__(self, db_url: Optional[str] = None, collectors: Optional[Dict[str, BaseCollector]] = None, kb=None):
        self.db_url = db_url
        self._collectors = collectors or {}
        self.kb = kb

    def collector(self, name: str) -> BaseCollector:
        if name not in self._collectors:
            self._collectors[name] = COLLECTORS[name]()
        return self._collectors[name]

    # ------------------------------------------------------------- storing
    def upsert(self, session, rec: PublicationRecord) -> tuple:
        """Store one record; returns (publication, created, matched_by)."""
        pub, matched_by = find_existing(session, rec)
        if pub is not None:
            merge_into(session, pub, rec)
            return pub, False, matched_by
        pub = new_publication(rec)
        session.add(pub)
        add_versions(session, pub, rec)
        session.flush()
        return pub, True, None

    def store_result(self, res: SearchResult, purpose: str, filters: Dict[str, Any],
                     query_kind: Optional[str] = None, assessment_id: Optional[int] = None) -> SourceOutcome:
        out = SourceOutcome(res.source, res.status, len(res.records), res.total_available, error=res.error)
        with session_scope(self.db_url) as s:
            run = SearchRun(purpose=purpose, source=res.source, query=res.query, query_kind=query_kind, filters=filters,
                            status=res.status, error=res.error, result_count=len(res.records),
                            total_available=res.total_available, novelty_assessment_id=assessment_id,
                            finished_at=datetime.now(timezone.utc))
            s.add(run)
            for rank, rec in enumerate(res.records, 1):
                pub, created, _ = self.upsert(s, rec)
                out.new += created
                out.merged += not created
                run.hits.append(SearchHit(rank=rank, publication_id=pub.id))
            s.flush()
            out.search_run_id = run.id
        return out

    # -------------------------------------------------------------- queries
    def collect(self, query: str, sources: Sequence[str] = DEFAULT_SOURCES, limit_per_source: int = 25,
                year_from: Optional[int] = None, year_to: Optional[int] = None, topic: Optional[str] = None,
                index_abstracts: bool = True) -> IngestReport:
        filters = {"year_from": year_from, "year_to": year_to, "limit_per_source": limit_per_source, "topic": topic}
        report = IngestReport(query=query, topic=topic, filters=filters)
        for name in sources:
            if name not in COLLECTORS:
                report.outcomes.append(SourceOutcome(name, "skipped", error="unknown source"))
                continue
            res = self.collector(name).search(query, limit=limit_per_source, year_from=year_from, year_to=year_to)
            outcome = self.store_result(res, "collection", filters)
            report.outcomes.append(outcome)
            if outcome.search_run_id:
                with session_scope(self.db_url) as s:
                    report.publication_ids += [h.publication_id for h in s.get(SearchRun, outcome.search_run_id).hits]
        report.finished_at = datetime.now(timezone.utc).isoformat()
        if index_abstracts and self.kb is not None:
            self.index_abstracts(report.publication_ids)
        return report

    def ingest_doi(self, doi: str) -> Optional[int]:
        """Look a DOI up in OpenAlex and Crossref and store the merged record."""
        pid = None
        for name in ("openalex", "crossref"):
            try:
                rec = self.collector(name).by_doi(doi)
            except Exception as exc:
                logger.warning(f"{name} DOI lookup failed: {exc}")
                rec = None
            res = SearchResult(name, doi, "ok" if rec else "failed", [rec] if rec else [],
                               error=None if rec else "not found or unavailable")
            out = self.store_result(res, "collection", {"doi": doi}, query_kind="doi")
            if rec and out.search_run_id:
                with session_scope(self.db_url) as s:
                    pid = s.get(SearchRun, out.search_run_id).hits[0].publication_id
        if pid and self.kb is not None:
            self.index_abstracts([pid])
        return pid

    def collect_author(self, author_name: str, limit: int = 25) -> IngestReport:
        report = IngestReport(query=author_name, topic=None, filters={"author": author_name})
        res = self.collector("openalex").by_author(author_name, limit=limit)
        out = self.store_result(res, "collection", {"author": author_name}, query_kind="author")
        report.outcomes.append(out)
        if out.search_run_id:
            with session_scope(self.db_url) as s:
                report.publication_ids = [h.publication_id for h in s.get(SearchRun, out.search_run_id).hits]
        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report

    # ------------------------------------------------------ knowledge base
    def index_abstracts(self, publication_ids: Iterable[int]) -> int:
        """Index title + abstract of each publication as one citable chunk. Metadata-only records are skipped."""
        if self.kb is None:
            return 0
        n = 0
        with session_scope(self.db_url) as s:
            for pub in s.scalars(select(Publication).where(Publication.id.in_(set(publication_ids)))):
                if not pub.abstract:
                    continue
                cid = f"pub{pub.id}-abstract"
                chunk = DocumentChunk(chunk_id=cid, text=f"{pub.title}. {pub.abstract}", source_filename=f"pub:{pub.id}",
                                      page_number=None, section_name="abstract", char_start=0,
                                      char_end=len(pub.abstract))
                n += self.kb.index_chunks([chunk], source_title=pub.title)
        return n

    def fetch_full_text(self, publication_id: int, doc_extractor=None) -> Dict[str, Any]:
        """Download and extract the open-access arXiv PDF of one publication (no other sources: no paywall bypass)."""
        from ..document_processing.document_extractor import DocumentExtractor
        with session_scope(self.db_url) as s:
            pub = s.get(Publication, publication_id)
            if pub is None:
                return {"ok": False, "reason": "unknown publication"}
            if not pub.arxiv_id:
                return {"ok": False, "reason": "Full text is only fetched from arXiv (open access); this record has no arXiv id."}
            arxiv_id, title = pub.arxiv_id, pub.title
        collector = self.collector("arxiv")
        collector._wait_turn()
        resp = collector.client.get(f"https://arxiv.org/pdf/{arxiv_id}")
        if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
            return {"ok": False, "reason": f"arXiv PDF unavailable (HTTP {resp.status_code})"}
        os.makedirs(FULLTEXT_DIR, exist_ok=True)
        path = os.path.join(FULLTEXT_DIR, f"{arxiv_id.replace('/', '_')}.pdf")
        with open(path, "wb") as f:
            f.write(resp.content)
        doc = (doc_extractor or DocumentExtractor()).extract_from_file(path)
        if not doc.full_text.strip():
            return {"ok": False, "reason": "PDF has no extractable text"}
        if self.kb is not None:
            chunks = TraceableChunker(chunk_size=600, chunk_overlap=100).chunk_document(
                doc.full_text, f"pub:{publication_id}", pages=doc.pages)
            self.kb.index_chunks(chunks, source_title=title)
        with session_scope(self.db_url) as s:
            pub = s.get(Publication, publication_id)
            pub.full_text_available = True
            pub.raw_metadata = {**(pub.raw_metadata or {}), "fulltext_path": os.path.relpath(path, ROOT),
                                "fulltext_pages": len(doc.pages),
                                "fulltext_retrieved_at": datetime.now(timezone.utc).isoformat()}
        return {"ok": True, "pages": len(doc.pages), "path": path}


# ------------------------------------------------------------------ read API
def list_publications(db_url: Optional[str] = None, text: Optional[str] = None, year_from: Optional[int] = None,
                      year_to: Optional[int] = None, source: Optional[str] = None, availability: Optional[str] = None,
                      include_synthetic: bool = False, limit: int = 500) -> List[Publication]:
    """availability: 'abstract' | 'metadata_only' | 'full_text' | None."""
    with session_scope(db_url) as s:
        q = select(Publication)
        if not include_synthetic:
            q = q.where(Publication.is_synthetic.is_(False))
        if text:
            like = f"%{text.lower()}%"
            q = q.where(or_(func.lower(Publication.title).like(like), func.lower(Publication.abstract).like(like)))
        if year_from:
            q = q.where(Publication.year >= year_from)
        if year_to:
            q = q.where(Publication.year <= year_to)
        if availability == "full_text":
            q = q.where(Publication.full_text_available.is_(True))
        elif availability == "abstract":
            q = q.where(Publication.abstract.is_not(None))
        elif availability == "metadata_only":
            q = q.where(Publication.abstract.is_(None))
        pubs = list(s.scalars(q.order_by(Publication.retrieved_at.desc()).limit(limit)))
        if source:
            pubs = [p for p in pubs if source in (p.raw_metadata or {}).get("sources", [p.source])]
        for p in pubs:
            _ = p.versions  # load before the session closes
        return pubs


def recent_search_runs(db_url: Optional[str] = None, purpose: str = "collection", limit: int = 30) -> List[SearchRun]:
    with session_scope(db_url) as s:
        return list(s.scalars(select(SearchRun).where(SearchRun.purpose == purpose)
                              .order_by(SearchRun.started_at.desc()).limit(limit)))
