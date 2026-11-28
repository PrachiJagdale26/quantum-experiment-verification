"""
Literature retrieval evaluation, measured on the known-work set.

Ground truth: each benchmark gap was written from a published paper's own question, so that paper is the one
relevant document we know of. The novelty benchmark stored every query it sent (SearchRun, kind and source) and
every hit with its rank (SearchHit), so retrieval is measured from those records; nothing is re-queried.

Metrics, per source and per query kind: Recall@k (the source paper is among the first k results), MRR (mean
reciprocal rank of the source paper, 0 when absent). "Any" pools all queries of a gap. k is bounded by the number
of results requested per query (10 in the benchmark), so Recall@10 is the maximum measurable.
Limitation: one known relevant paper per query; other relevant papers in the results are not judged, so
precision is not measured (it needs the topic set with relevance judgements).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..storage import CandidateGap, NoveltyAssessment, SearchRun, session_scope

KS = (1, 3, 10)


def _summ(ranks: List[Optional[int]]) -> Dict[str, Any]:
    n = len(ranks)
    out = {"n": n, "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else None}
    for k in KS:
        out[f"recall@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3) if n else None
    return out


def evaluate_retrieval(db_url: Optional[str]) -> Dict[str, Any]:
    with session_scope(db_url) as s:
        gaps = [g for g in s.scalars(select(CandidateGap).where(CandidateGap.origin == "benchmark"))
                if (g.payload or {}).get("benchmark", {}).get("source_publication_id")]
        by_source: Dict[str, List[Optional[int]]] = defaultdict(list)
        by_kind: Dict[str, List[Optional[int]]] = defaultdict(list)
        pooled: List[Optional[int]] = []
        failed_runs = 0
        for g in gaps:
            src = g.payload["benchmark"]["source_publication_id"]
            a = s.scalars(select(NoveltyAssessment).where(NoveltyAssessment.gap_id == g.id,
                                                          NoveltyAssessment.classification != "pending")
                          .order_by(NoveltyAssessment.id.desc())).first()
            if a is None:
                continue
            runs = list(s.scalars(select(SearchRun).where(SearchRun.novelty_assessment_id == a.id,
                                                          SearchRun.purpose == "novelty")))
            best_src: Dict[str, Optional[int]] = {}
            best_kind: Dict[str, Optional[int]] = {}
            for r in runs:
                if r.status not in ("ok", "cached"):
                    failed_runs += 1
                    continue
                rank = next((h.rank for h in r.hits if h.publication_id == src), None)
                for d, key in ((best_src, r.source), (best_kind, r.query_kind or "?")):
                    if rank and (d.get(key) is None or rank < d[key]):
                        d[key] = rank
                    d.setdefault(key, None)
            for k, v in best_src.items():
                by_source[k].append(v)
            for k, v in best_kind.items():
                by_kind[k].append(v)
            ranks = [v for v in best_src.values() if v]
            pooled.append(min(ranks) if ranks else None)
    return {"procedure": __doc__.strip(), "gaps": len(pooled), "failed_or_rate_limited_searches": failed_runs,
            "any_query_any_source": _summ(pooled),
            "by_source": {k: _summ(v) for k, v in sorted(by_source.items())},
            "by_query_kind": {k: _summ(v) for k, v in sorted(by_kind.items())}}
