"""
Candidate gap generation from the curated, analysed corpus (Module 4C).

Sources of candidates (deterministic; no model needed):
  1. Verified limitation / unresolved-problem / future-work statements. Statements that say the same thing
     across papers are merged, so one candidate carries every paper that raises it.
  2. Untried combinations in the corpus (method x platform, method x noise) where both sides are each studied
     by 2+ papers. These are inferences about THIS corpus only; novelty is decided later by searching.
Unverified model statements are never used. An optional model pass rewrites each gap as a research question;
the rewrite is labelled with the model, and the original evidence is left untouched.
"""

import hashlib
import logging
from typing import Callable, Dict, List, Optional, Sequence

from sqlalchemy import select

from ..knowledge_base.ontology import concepts_by_category
from ..llm.json_utils import parse_llm_json
from ..research_analysis.cross_paper_analysis import _clusters, _untried, load_corpus
from ..research_analysis.paper_analyzer import normalize_for_match, qubit_counts
from ..research_analysis.research_schema import PaperAnalysisRecord, ResearchDimensions
from ..storage import CandidateGap, GapEvidence, Publication, session_scope
from .schema import Gap, GapDimension, GapEvidenceItem

logger = logging.getLogger(__name__)

FIELD_TO_ORIGIN = {"limitation": "stated_limitation", "unresolved_problem": "unresolved_problem", "future_work": "future_work"}
CAT_TO_ATTR = {"phenomenon": "phenomena", "method": "methods", "platform": "platforms", "noise": "noise", "metric": "metrics"}
VAGUE = {"hardware noise (unspecified)"}           # too unspecific to define a research question
MAX_CONTEXT_VALUES = 2      # dimensions borrowed from a long source paper would otherwise match almost anything

GAP_QUESTION_PROMPT = """You rewrite candidate research gaps as precise research questions.
Rules: use ONLY information in the gap text and its quoted evidence. Do not add systems, numbers, methods or claims
that are not there. One sentence each, ending with a question mark. Return JSON only:
{"items": [{"key": "<key>", "question": "<research question>"}]}"""


def dimensions_for(text: str, paper: Optional[ResearchDimensions]) -> Dict[str, GapDimension]:
    """Dimensions named in the gap's own text first; the source paper's dimensions only fill what the text leaves out."""
    cats = concepts_by_category(text)
    dims: Dict[str, GapDimension] = {}
    for cat, attr in CAT_TO_ATTR.items():
        own = [c for c in cats.get(cat, []) if c not in VAGUE]
        ctx = [c for c in (getattr(paper, attr) if paper else []) if c not in VAGUE][:MAX_CONTEXT_VALUES]
        if own:
            dims[cat] = GapDimension(values=own, source="statement")
        elif ctx:
            dims[cat] = GapDimension(values=ctx, source="paper_context")
    own_q, ctx_q = qubit_counts(text), (paper.qubit_counts if paper else [])
    if own_q or ctx_q:
        dims["qubit_scale"] = GapDimension(values=[str(q) for q in (own_q or ctx_q)], source="statement" if own_q else "paper_context")
        dims["qubit_scale"].values = dims["qubit_scale"].values[:MAX_CONTEXT_VALUES] if not own_q else dims["qubit_scale"].values
    return dims


def _key(*parts: str) -> str:
    return hashlib.sha1(normalize_for_match(" | ".join(parts)).encode()).hexdigest()[:16]


def _short(text: str, words: int = 14) -> str:
    w = text.split()
    return " ".join(w[:words]).rstrip(",;:") + ("…" if len(w) > words else "")


class GapGenerator:
    def __init__(self, db_url: Optional[str] = None, embed: Optional[Callable] = None, llm=None,
                 merge_threshold: float = 0.62, min_support: int = 2):
        self.db_url, self._embed, self.llm = db_url, embed, llm
        self.merge_threshold, self.min_support = merge_threshold, min_support

    @property
    def embed(self):
        if self._embed is None:
            from ..research_analysis.cross_paper_analysis import default_embedder
            self._embed = default_embedder()
        return self._embed

    # ---------------------------------------------------------------- generation
    def generate(self, records: Optional[Sequence[PaperAnalysisRecord]] = None) -> List[Gap]:
        if records is None:
            records, _, _ = load_corpus(self.db_url)
        gaps = self._from_statements(records) + self._from_combinations(records)
        if self.llm is not None and not getattr(self.llm, "demo_mode", True):
            self._rewrite_questions(gaps)
        return gaps

    def _from_statements(self, records: Sequence[PaperAnalysisRecord]) -> List[Gap]:
        gaps: List[Gap] = []
        for fld, origin in FIELD_TO_ORIGIN.items():
            items = [(r, s) for r in records for s in r.statements
                     if s.field == fld and s.kind in ("direct", "interpretation") and s.provenance and s.provenance.verified]
            if not items:
                continue
            groups = _clusters(self.embed([s.text for _, s in items]), self.merge_threshold) if len(items) > 1 else [[0]]
            for idx in groups:
                members = sorted((items[i] for i in idx), key=lambda rs: (rs[1].kind != "direct", -rs[1].confidence))
                rep_rec, rep = members[0]
                evidence = [GapEvidenceItem(publication_id=r.publication_id, title=r.title,
                                            quote=s.provenance.excerpt if s.kind == "interpretation" else s.text,
                                            kind=s.kind, page=s.provenance.page) for r, s in members]
                papers = {r.publication_id for r, _ in members}
                gaps.append(Gap(
                    key=_key(origin, str(rep_rec.publication_id), rep.text), origin=origin,
                    title=_short(rep.text), statement=rep.text,
                    dimensions=dimensions_for(" ".join(s.text for _, s in members), rep_rec.dimensions),
                    evidence=evidence, generator="deterministic" + (f" (merged from {len(papers)} papers)" if len(papers) > 1 else "")))
        return gaps

    def _from_combinations(self, records: Sequence[PaperAnalysisRecord]) -> List[Gap]:
        gaps: List[Gap] = []
        for col_attr, cat in (("platforms", "platform"), ("noise", "noise")):
            for u in _untried(records, "methods", col_attr, self.min_support):
                if u["col"] in VAGUE:
                    continue
                with_row = [r for r in records if u["row"] in r.dimensions.methods]
                with_col = [r for r in records if u["col"] in getattr(r.dimensions, col_attr)]
                link = "on" if cat == "platform" else "under"
                statement = (f"No paper in this corpus applies {u['row']} {link} {u['col']}, although "
                             f"{u['row_papers']} papers use {u['row']} and {u['col_papers']} papers study {u['col']}.")
                evidence = ([GapEvidenceItem(publication_id=r.publication_id, title=r.title, quote=f"Uses {u['row']}: {r.title}",
                                             kind="inference", role="related") for r in with_row[:3]]
                            + [GapEvidenceItem(publication_id=r.publication_id, title=r.title, quote=f"Studies {u['col']}: {r.title}",
                                               kind="inference", role="related") for r in with_col[:3]])
                gaps.append(Gap(
                    key=_key("unexplored_combination", u["row"], u["col"]), origin="unexplored_combination",
                    title=f"{u['row'].capitalize()} {link} {u['col']}", statement=statement,
                    dimensions={"method": GapDimension(values=[u["row"]], source="corpus"),
                                cat: GapDimension(values=[u["col"]], source="corpus")},
                    evidence=evidence))
        return gaps

    def _rewrite_questions(self, gaps: List[Gap], batch: int = 20) -> None:
        for i in range(0, len(gaps), batch):
            chunk = gaps[i:i + batch]
            payload = "\n".join(f"- key: {g.key}\n  gap: {g.statement}\n  evidence: {g.evidence[0].quote if g.evidence else ''}"
                                for g in chunk)
            try:
                raw = self.llm.complete(GAP_QUESTION_PROMPT, payload, fallback_to_demo=False, purpose="gap_questions")
                items = (parse_llm_json(raw) or {}).get("items", [])
            except Exception as exc:           # a failed rewrite leaves the deterministic statement in place
                logger.warning("research-question rewrite failed: %s", exc)
                return
            model = "{provider}/{model}".format(**getattr(self.llm, "last_call", {}) or {"provider": "llm", "model": "?"})
            by_key = {g.key: g for g in chunk}
            for it in items:
                g = by_key.get(str(it.get("key")))
                q = str(it.get("question") or "").strip()
                if g and q.endswith("?"):
                    g.research_question, g.research_question_source = q, model


# ------------------------------------------------------------------- persistence
def _to_payload(g: Gap, old: Optional[dict] = None) -> dict:
    out = dict(old or {})
    out.update(key=g.key, research_question=g.research_question, research_question_source=g.research_question_source,
               dimensions={k: v.model_dump() for k, v in g.dimensions.items()})
    return out


def save_gaps(gaps: Sequence[Gap], db_url: Optional[str] = None) -> List[Gap]:
    """Insert new candidates; update existing ones (same key) without touching their assessments or review status."""
    with session_scope(db_url) as s:
        existing = {(c.payload or {}).get("key"): c for c in s.scalars(select(CandidateGap))}
        for g in gaps:
            row = existing.get(g.key)
            if row is None:
                row = CandidateGap(title=g.title, statement=g.statement, origin=g.origin, generator=g.generator, payload={})
                s.add(row)
            row.title, row.statement, row.generator = g.title, g.statement, g.generator
            row.payload = _to_payload(g, row.payload)
            row.evidence.clear()
            for e in g.evidence:
                row.evidence.append(GapEvidence(publication_id=e.publication_id, role=e.role, quote=e.quote, page=e.page, kind=e.kind))
            s.flush()
            g.id, g.classification, g.review_status = row.id, row.classification, row.review_status
    return list(gaps)


def load_gaps(db_url: Optional[str] = None, include_benchmark: bool = False) -> List[Gap]:
    with session_scope(db_url) as s:
        q = select(CandidateGap).order_by(CandidateGap.id)
        if not include_benchmark:
            q = q.where(CandidateGap.origin != "benchmark")
        rows = list(s.scalars(q))
        titles = {p.id: p.title for p in s.scalars(select(Publication).where(
            Publication.id.in_({e.publication_id for r in rows for e in r.evidence if e.publication_id})))}
        return [row_to_gap(r, titles) for r in rows]


def load_gap(gap_id: int, db_url: Optional[str] = None) -> Optional[Gap]:
    with session_scope(db_url) as s:
        r = s.get(CandidateGap, gap_id)
        if r is None:
            return None
        titles = {p.id: p.title for p in s.scalars(select(Publication).where(
            Publication.id.in_({e.publication_id for e in r.evidence if e.publication_id})))}
        return row_to_gap(r, titles)


def row_to_gap(r: CandidateGap, titles: Dict[int, str]) -> Gap:
    p = r.payload or {}
    return Gap(id=r.id, key=p.get("key") or f"row-{r.id}", title=r.title, statement=r.statement, origin=r.origin,
               research_question=p.get("research_question"), research_question_source=p.get("research_question_source"),
               dimensions={k: GapDimension(**v) for k, v in (p.get("dimensions") or {}).items()},
               evidence=[GapEvidenceItem(publication_id=e.publication_id, title=titles.get(e.publication_id, "Unknown paper"),
                                         quote=e.quote, kind=e.kind if e.kind in ("direct", "interpretation", "inference") else "inference",
                                         page=e.page, role=e.role if e.role in ("supporting", "related") else "related")
                         for e in r.evidence],
               generator=r.generator, classification=r.classification, review_status=r.review_status)
