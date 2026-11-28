"""
Duplicate detection and merging for publication records from several sources.

Matching order (first hit wins):
  1. DOI (normalised)            2. arXiv id (version-less)      3. OpenAlex id
  4. Fuzzy title: token-sort ratio >= TITLE_THRESHOLD, year within +/-1, and the same first-author
     surname when both records have authors.
Merging fills empty fields only; it never overwrites a value with a different one silently. The
journal DOI is preferred over an arXiv DataCite DOI (10.48550/arxiv.*).
"""

from typing import Optional, Tuple

from rapidfuzz import fuzz
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..storage import Publication, PublicationVersion
from .collectors.base import PublicationRecord
from .metadata import first_author_surname, normalize_title

TITLE_THRESHOLD = 93
ARXIV_DOI_PREFIX = "10.48550/arxiv."


def _is_arxiv_doi(doi: Optional[str]) -> bool:
    return bool(doi and doi.startswith(ARXIV_DOI_PREFIX))


def titles_match(a_title: str, a_year: Optional[int], a_authors, b_title: str, b_year: Optional[int], b_authors) -> bool:
    na, nb = normalize_title(a_title), normalize_title(b_title)
    if not na or not nb or fuzz.token_sort_ratio(na, nb) < TITLE_THRESHOLD:
        return False
    if a_year and b_year and abs(a_year - b_year) > 1:
        return False
    sa, sb = first_author_surname(a_authors), first_author_surname(b_authors)
    return not (sa and sb and sa != sb)


def find_existing(session: Session, rec: PublicationRecord) -> Tuple[Optional[Publication], Optional[str]]:
    """Return (publication, matched_by) for an already-stored work, or (None, None)."""
    for attr, value in (("doi", rec.doi), ("arxiv_id", rec.arxiv_id), ("openalex_id", rec.openalex_id)):
        if value:
            pub = session.scalars(select(Publication).where(getattr(Publication, attr) == value)).first()
            if pub:
                return pub, attr
    # arXiv DataCite DOIs encode the arXiv id: 10.48550/arxiv.2303.04792
    if _is_arxiv_doi(rec.doi) and not rec.arxiv_id:
        pub = session.scalars(select(Publication).where(Publication.arxiv_id == rec.doi[len(ARXIV_DOI_PREFIX):])).first()
        if pub:
            return pub, "arxiv_doi"
    norm = normalize_title(rec.title)
    if not norm:
        return None, None
    q = select(Publication)
    if rec.year:
        q = q.where(or_(Publication.year.is_(None), Publication.year.between(rec.year - 1, rec.year + 1)))
    for pub in session.scalars(q):
        if _different_identifiers(pub, rec):
            continue  # different DOIs / arXiv ids mean different works, however similar the titles
        if titles_match(rec.title, rec.year, rec.authors, pub.title, pub.year, pub.authors):
            return pub, "fuzzy_title"
    return None, None


def _different_identifiers(pub: Publication, rec: PublicationRecord) -> bool:
    if pub.doi and rec.doi and pub.doi != rec.doi and not (_is_arxiv_doi(pub.doi) or _is_arxiv_doi(rec.doi)):
        return True
    return bool(pub.arxiv_id and rec.arxiv_id and pub.arxiv_id != rec.arxiv_id)


def _id_free(session: Session, attr: str, value: str, pub: Publication) -> bool:
    other = session.scalars(select(Publication).where(getattr(Publication, attr) == value)).first()
    return other is None or other.id == pub.id


def merge_into(session: Session, pub: Publication, rec: PublicationRecord) -> None:
    """Fill gaps in `pub` from `rec`; record the source and any conflicting identifiers."""
    meta = dict(pub.raw_metadata or {})
    sources = set(meta.get("sources", [pub.source]))
    sources.add(rec.source)
    meta["sources"] = sorted(sources)
    conflicts = meta.setdefault("identifier_conflicts", [])

    def set_id(attr: str, value: Optional[str]):
        if not value:
            return
        current = getattr(pub, attr)
        if current == value:
            return
        if current and not (attr == "doi" and _is_arxiv_doi(current) and not _is_arxiv_doi(value)):
            if attr == "doi" and _is_arxiv_doi(value):
                meta["arxiv_doi"] = value
            else:
                conflicts.append({"field": attr, "kept": current, "other": value, "source": rec.source})
            return
        if _id_free(session, attr, value, pub):
            if attr == "doi" and current:
                meta["arxiv_doi"] = current
            setattr(pub, attr, value)
        else:
            conflicts.append({"field": attr, "value": value, "reason": "held by another record", "source": rec.source})

    set_id("doi", rec.doi)
    set_id("arxiv_id", rec.arxiv_id)
    set_id("openalex_id", rec.openalex_id)

    if not pub.abstract and rec.abstract:
        pub.abstract, pub.abstract_source = rec.abstract, rec.source
    if rec.venue and (not pub.venue or pub.venue == "arXiv preprint"):
        pub.venue = rec.venue
    if not pub.year and rec.year:
        pub.year = rec.year
    if not pub.publication_date and rec.publication_date:
        pub.publication_date = rec.publication_date
    if not pub.authors and rec.authors:
        pub.authors = rec.authors
    if not pub.pdf_url and rec.pdf_url:
        pub.pdf_url = rec.pdf_url
    pub.is_open_access = bool(pub.is_open_access or rec.is_open_access)
    per_source = dict(meta.get("by_source", {}))
    per_source[rec.source] = {"source_url": rec.source_url, "retrieved_at": rec.retrieved_at,
                              "cited_by_count": rec.cited_by_count, **{k: v for k, v in rec.raw.items() if v}}
    meta["by_source"] = per_source
    if rec.referenced_works and not meta.get("referenced_works"):
        meta["referenced_works"] = rec.referenced_works
    pub.raw_metadata = meta
    add_versions(session, pub, rec)


def add_versions(session: Session, pub: Publication, rec: PublicationRecord) -> None:
    existing = {v.version_label for v in pub.versions}
    for v in rec.versions:
        label = f"arXiv {v['label']}" if rec.source == "arxiv" else v["label"]
        if label not in existing:
            pub.versions.append(PublicationVersion(version_label=label, version_date=v.get("date"),
                                                   source_url=rec.source_url))
            existing.add(label)
    if rec.doi and not _is_arxiv_doi(rec.doi) and rec.source in ("crossref", "openalex") and "published version" not in existing:
        pub.versions.append(PublicationVersion(version_label="published version", version_date=rec.publication_date,
                                               source_url=f"https://doi.org/{rec.doi}"))


def new_publication(rec: PublicationRecord) -> Publication:
    pub = Publication(
        doi=rec.doi, arxiv_id=rec.arxiv_id, openalex_id=rec.openalex_id, title=rec.title,
        normalized_title=normalize_title(rec.title), year=rec.year, publication_date=rec.publication_date,
        venue=rec.venue, authors=rec.authors, abstract=rec.abstract,
        abstract_source=rec.source if rec.abstract else None, source=rec.source, source_url=rec.source_url,
        pdf_url=rec.pdf_url, is_open_access=rec.is_open_access, full_text_available=False,
        raw_metadata={"sources": [rec.source], "by_source": {rec.source: {
            "source_url": rec.source_url, "retrieved_at": rec.retrieved_at, "cited_by_count": rec.cited_by_count,
            **{k: v for k, v in rec.raw.items() if v}}},
            **({"referenced_works": rec.referenced_works} if rec.referenced_works else {})},
    )
    if _is_arxiv_doi(rec.doi) and rec.arxiv_id:
        pub.raw_metadata["arxiv_doi"] = rec.doi
    return pub
