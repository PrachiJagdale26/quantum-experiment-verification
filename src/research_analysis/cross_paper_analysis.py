"""
Cross-paper analysis (Module 4B) over the curated corpus.

Inputs: stored analyses of publications that are not excluded and not synthetic. Only verified
statements (direct quotes or verified interpretations) are used; unverified model output never is.
Every output here is an INFERENCE across papers and says which papers support it.

- themes: papers clustered by meaning (embeddings of title + objectives), labelled with ontology concepts
- coverage: method x platform and method x noise matrices; empty cells next to well-studied rows/columns are
  "untried in this corpus" (absence in the collected papers, not evidence of absence in the literature)
- recurring statements: limitations / unresolved problems / future work / assumptions stated by 2+ papers
"""

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import select

from ..storage import Publication, session_scope
from .paper_analyzer import load_analysis
from .research_schema import PaperAnalysisRecord

GENERIC = {"hardware noise (unspecified)", "entanglement", "fidelity"}   # too common to name a theme on their own
RECURRING_FIELDS = ("limitation", "unresolved_problem", "future_work", "assumption")


@dataclass
class Theme:
    label: str
    concepts: List[str]
    publication_ids: List[int]
    titles: List[str]


@dataclass
class CoverageCell:
    row: str
    col: str
    publication_ids: List[int]


@dataclass
class RecurringStatement:
    field: str
    representative: str
    members: List[Dict]          # {publication_id, title, text, kind, page}
    papers: int


@dataclass
class CorpusAnalysis:
    papers: int
    excluded: int
    not_analysed: int
    themes: List[Theme] = field(default_factory=list)
    method_platform: List[CoverageCell] = field(default_factory=list)
    method_noise: List[CoverageCell] = field(default_factory=list)
    untried_in_corpus: List[Dict] = field(default_factory=list)
    recurring: List[RecurringStatement] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict:
        return asdict(self)


def default_embedder() -> Callable[[List[str]], np.ndarray]:
    from ..knowledge_base.vector_store import _load_embedding_model
    import os
    model = _load_embedding_model(os.getenv("EMBEDDING_MODEL_NAME") or "all-MiniLM-L6-v2")
    return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True))


def load_corpus(db_url: Optional[str] = None) -> tuple:
    """(analysed records in scope, excluded count, in-scope but not analysed count)."""
    with session_scope(db_url) as s:
        pubs = list(s.scalars(select(Publication).where(Publication.is_synthetic.is_(False))))
        excluded = sum(1 for p in pubs if p.curation_status == "excluded")
        scope = [(p.id, p.year) for p in pubs if p.curation_status != "excluded"]
        not_analysed = sum(1 for p in pubs if p.curation_status != "excluded" and p.analysed_at is None)
    records = [r for r in (load_analysis(pid, db_url) for pid, _ in scope) if r is not None]
    return records, excluded, not_analysed


def _clusters(vectors: np.ndarray, threshold: float) -> List[List[int]]:
    """
    Average-linkage agglomerative clustering on cosine distance, cut at 1 - threshold. Deterministic and needs
    no number of clusters. (Single linkage / connected components was rejected: it chains a narrow-topic
    corpus into one giant cluster.)
    """
    n = len(vectors)
    if n == 1:
        return [[0]]
    from sklearn.cluster import AgglomerativeClustering
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1.0 - threshold).fit_predict(vectors)
    groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


_TITLE_STOP = set("the and for with from into under over about using based study analysis quantum theory effects effect "
                  "towards toward via new between their its this that system systems".split())


def _distinctive_title_words(cluster_titles: List[str], corpus_titles: List[str], k: int = 3) -> List[str]:
    tok = lambda t: {w for w in re.findall(r"[a-z][a-z\-]{3,}", t.lower()) if w not in _TITLE_STOP}
    corpus = Counter(w for t in corpus_titles for w in tok(t))
    local = Counter(w for t in cluster_titles for w in tok(t))
    n_c, n_all = len(cluster_titles), len(corpus_titles)
    scored = [(local[w] / n_c - corpus[w] / n_all, w) for w in local if local[w] >= max(2, (n_c + 1) // 2)]
    return [w for s, w in sorted(scored, reverse=True) if s > 0][:k]


def _themes(records: Sequence[PaperAnalysisRecord], embed, threshold: float) -> List[Theme]:
    if len(records) < 2:
        return []
    texts = [r.title + ". " + " ".join(s.text for s in r.statements if s.field in ("objective", "research_problem"))[:600]
             for r in records]
    concepts_of = lambda r: set(r.dimensions.methods + r.dimensions.phenomena + r.dimensions.platforms + r.dimensions.noise)
    corpus_share = {c: n / len(records) for c, n in Counter(c for r in records for c in concepts_of(r)).items()}
    themes = []
    for idx in _clusters(embed(texts), threshold):
        members = [records[i] for i in idx]
        share = {c: n / len(members) for c, n in Counter(c for r in members for c in concepts_of(r)).items()}
        # Label by what is distinctive: concepts over-represented here relative to the corpus, skipping
        # concepts that most of the corpus shares (e.g. the collection's own search topic).
        distinctive = sorted((c for c in share if corpus_share[c] < 0.6 and c not in GENERIC and share[c] >= 0.5),
                             key=lambda c: (share[c] - corpus_share[c], share[c]), reverse=True)
        concepts = distinctive[:3]
        if concepts:
            label = ", ".join(concepts)
        else:
            # The ontology covers quantum computing; for other areas fall back to distinctive title words.
            words = _distinctive_title_words([r.title for r in members], [r.title for r in records])
            label = ("Keywords: " + ", ".join(words)) if words else "General (no distinctive concept)"
        themes.append(Theme(label=label if len(members) > 1 else f"{label} (single paper)", concepts=concepts,
                            publication_ids=[r.publication_id for r in members], titles=[r.title for r in members]))
    return themes


def _matrix(records: Sequence[PaperAnalysisRecord], row_attr: str, col_attr: str) -> List[CoverageCell]:
    cells: Dict[tuple, List[int]] = {}
    for r in records:
        for row in getattr(r.dimensions, row_attr):
            for col in getattr(r.dimensions, col_attr):
                cells.setdefault((row, col), []).append(r.publication_id)
    return [CoverageCell(row, col, sorted(set(p))) for (row, col), p in sorted(cells.items())]


def _untried(records: Sequence[PaperAnalysisRecord], row_attr: str, col_attr: str, min_support: int) -> List[Dict]:
    row_n = Counter(x for r in records for x in set(getattr(r.dimensions, row_attr)))
    col_n = Counter(x for r in records for x in set(getattr(r.dimensions, col_attr)))
    seen = {(c.row, c.col) for c in _matrix(records, row_attr, col_attr)}
    out = []
    for row, rn in row_n.items():
        for col, cn in col_n.items():
            if rn >= min_support and cn >= min_support and (row, col) not in seen:
                out.append({"row": row, "col": col, "row_papers": rn, "col_papers": cn,
                            "kind": "inference", "note": "Not studied together in this corpus; novelty is not established."})
    return sorted(out, key=lambda d: -(d["row_papers"] + d["col_papers"]))


def _recurring(records: Sequence[PaperAnalysisRecord], embed, threshold: float) -> List[RecurringStatement]:
    out: List[RecurringStatement] = []
    for fld in RECURRING_FIELDS:
        items = [(r, s) for r in records for s in r.statements
                 if s.field == fld and s.provenance and s.provenance.verified and s.kind in ("direct", "interpretation")]
        if len(items) < 2:
            continue
        for idx in _clusters(embed([s.text for _, s in items]), threshold):
            papers = {items[i][0].publication_id for i in idx}
            if len(papers) < 2:
                continue
            members = [{"publication_id": items[i][0].publication_id, "title": items[i][0].title, "text": items[i][1].text,
                        "kind": items[i][1].kind, "page": items[i][1].provenance.page} for i in idx]
            rep = max((items[i][1] for i in idx), key=lambda s: (s.kind == "direct", s.confidence)).text
            out.append(RecurringStatement(field=fld, representative=rep, members=members, papers=len(papers)))
    return sorted(out, key=lambda r: -r.papers)


def analyse_corpus(db_url: Optional[str] = None, embed: Optional[Callable] = None, theme_threshold: float = 0.55,
                   statement_threshold: float = 0.62, min_support: int = 2) -> CorpusAnalysis:
    records, excluded, not_analysed = load_corpus(db_url)
    result = CorpusAnalysis(papers=len(records), excluded=excluded, not_analysed=not_analysed)
    if not records:
        return result
    embed = embed or default_embedder()
    result.themes = _themes(records, embed, theme_threshold)
    result.method_platform = _matrix(records, "methods", "platforms")
    result.method_noise = _matrix(records, "methods", "noise")
    result.untried_in_corpus = (_untried(records, "methods", "platforms", min_support)
                                + _untried(records, "methods", "noise", min_support))[:20]
    result.recurring = _recurring(records, embed, statement_threshold)
    return result
