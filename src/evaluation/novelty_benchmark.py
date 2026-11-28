"""
Known-work novelty benchmark: the main evaluation of Phase 4.

Procedure (ground truth is objective: the work IS published):
  1. Take analysed, real (non-synthetic, not excluded, collected) publications with an abstract.
  2. Turn each paper's own verified research question / objective / problem statement into a candidate gap.
  3. Remove that paper from the local corpus for this check, and run the full novelty investigation
     (external search, citations, comparison, rule-based classification).
  4. A "potentially unexplored" verdict is a FALSE NOVELTY: the question is published.

Metrics: false-novelty rate, detection rate (already or partially investigated), insufficient-coverage rate,
recall of the source paper among screened publications and among the compared top-k, and MRR of the source paper.
Every run writes its full procedure, configuration, per-item results and limitations to JSON.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select

from ..gap_discovery.gap_generator import dimensions_for, save_gaps
from ..gap_discovery.novelty_checker import NoveltyChecker
from ..gap_discovery.schema import Gap, GapDimension
from ..research_analysis.paper_analyzer import load_analysis
from ..storage import CandidateGap, Publication, session_scope

QUESTION_FIELDS = ("research_question", "objective", "research_problem")

LIMITATIONS = [
    "The gap text is the paper's own sentence, so the exact-keyword query shares wording with the target; "
    "this makes retrieval easier than for a gap phrased independently. Recall is therefore an upper bound.",
    "Dimensions missing from the sentence are filled from the source paper's own tags (as for real gaps).",
    "The source paper stays in the external indexes; only the local corpus excludes it.",
    "Comparisons use titles and abstracts; papers without abstracts are compared on titles.",
    "Thresholds in CoveragePolicy were set before this run; tuning them on this set would overfit.",
    "A small set: rates have wide uncertainty (Wilson 95% interval reported).",
    "If most questions share the corpus topic (see distinct_core_dimension_sets), the set cannot separate a good "
    "checker from one that always answers 'already investigated'; the specificity control is reported for that reason.",
    "Ontology surface forms are matched without word sense: 'leakage' also matches privacy leakage, and "
    "'integrated photonics' marks a trapped-ion device as photonic.",
]


# Specificity control. No ground truth: these combinations were chosen as implausible, and the last one is a
# well-known positive control. A checker that answers "already investigated" to everything fails this control,
# which the known-work set alone cannot show when its questions share the corpus topic.
CONTROLS = [
    {"phenomenon": "barren plateaus", "platform": "nv centers", "noise": "leakage"},
    {"phenomenon": "measurement-induced phase transitions", "platform": "photonic qubits", "noise": "crosstalk"},
    {"phenomenon": "barren plateaus", "platform": "topological qubits", "noise": "non-markovian noise"},
    {"method": "quantum machine learning", "platform": "nv centers", "noise": "leakage"},
    {"method": "tomography", "platform": "topological qubits", "noise": "non-markovian noise"},
    {"method": "quantum error correction", "platform": "trapped ions", "noise": "depolarizing noise", "positive": True},
]


def run_controls(db_url: Optional[str], checker: NoveltyChecker) -> List[dict]:
    out = []
    for c in CONTROLS:
        dims = {k: GapDimension(values=[v], source="statement") for k, v in c.items() if k != "positive"}
        lead = c.get("phenomenon") or c.get("method")
        st = f"Study {lead} on {c['platform']} under {c['noise']}."
        g = Gap(key=f"control:{st}", origin="benchmark", title=st, statement=st, dimensions=dims, generator="control")
        save_gaps([g], db_url)
        a = checker.check(g.id)
        top = (a["findings"]["closest"] or [None])[0]
        out.append({"gap": st, "positive_control": bool(c.get("positive")), "classification": a["classification"],
                    "confidence": a["confidence"],
                    "closest": {"title": top["title"], "same": top["same"], "similarity": top["similarity"]} if top else None})
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def benchmark_items(db_url: Optional[str], n: int) -> List[dict]:
    with session_scope(db_url) as s:
        pids = [p.id for p in s.scalars(select(Publication).where(
            Publication.discovered_via == "collection", Publication.is_synthetic.is_(False),
            Publication.curation_status != "excluded", Publication.analysed_at.is_not(None),
            Publication.abstract.is_not(None)).order_by(Publication.id))]
    items = []
    for pid in pids:
        rec = load_analysis(pid, db_url)
        stmts = [st for f in QUESTION_FIELDS for st in rec.statements
                 if st.field == f and st.kind in ("direct", "interpretation") and st.provenance and st.provenance.verified]
        if not stmts:
            continue
        items.append({"publication_id": pid, "title": rec.title, "question": stmts[0].text, "field": stmts[0].field,
                      "record": rec})
        if len(items) >= n:
            break
    return items


def run_known_work(db_url: Optional[str], n: int = 10, checker: Optional[NoveltyChecker] = None,
                   out_dir: Optional[str] = None, llm_label: Optional[str] = None) -> Dict:
    checker = checker or NoveltyChecker(db_url)
    started = datetime.now(timezone.utc)
    results = []
    for it in benchmark_items(db_url, n):
        pid = it["publication_id"]
        g = Gap(key=f"benchmark:{pid}", origin="benchmark", title=f"Benchmark: {it['title'][:80]}", statement=it["question"],
                dimensions=dimensions_for(it["question"], it["record"].dimensions), generator="benchmark")
        save_gaps([g], db_url)
        with session_scope(db_url) as s:
            row = s.get(CandidateGap, g.id)
            row.payload = {**(row.payload or {}), "benchmark": {"source_publication_id": pid, "field": it["field"]}}
        a = checker.check(g.id, exclude_local_ids={pid})
        closest_ids = [c["publication_id"] for c in a["findings"]["closest"]]
        rank = closest_ids.index(pid) + 1 if pid in closest_ids else None
        results.append({"publication_id": pid, "title": it["title"], "question": it["question"], "field": it["field"],
                        "core_dimensions": a["findings"]["core_dimensions"], "classification": a["classification"],
                        "confidence": a["confidence"], "false_novelty": a["classification"] == "potentially_unexplored",
                        "found_in_screened": pid in a["coverage"]["screened_ids"], "rank_in_compared": rank,
                        "decided_by_source_paper": a["findings"]["decided_by"] == pid,
                        "sources_ok": a["coverage"]["sources_ok"], "screened": a["coverage"]["screened"],
                        "queries": [q["query"] for q in a["coverage"]["queries"]]})
    controls = run_controls(db_url, checker)
    m = len(results)
    count = lambda c: sum(1 for r in results if r["classification"] == c)
    fn = sum(r["false_novelty"] for r in results)
    detected = count("already_investigated") + count("partially_investigated")
    metrics = {
        "items": m,
        "false_novelty_rate": round(fn / m, 3) if m else None, "false_novelty_ci95": _wilson(fn, m),
        "detection_rate": round(detected / m, 3) if m else None,
        "insufficient_coverage_rate": round(count("insufficient_coverage") / m, 3) if m else None,
        "by_classification": {c: count(c) for c in ("already_investigated", "partially_investigated", "conflicting_findings",
                                                    "potentially_unexplored", "insufficient_coverage")},
        "recall_screened": round(sum(r["found_in_screened"] for r in results) / m, 3) if m else None,
        "recall_compared_top_k": round(sum(r["rank_in_compared"] is not None for r in results) / m, 3) if m else None,
        "mrr_compared": round(sum(1 / r["rank_in_compared"] for r in results if r["rank_in_compared"]) / m, 3) if m else None,
        "decided_by_source_paper": sum(r["decided_by_source_paper"] for r in results),
        "distinct_core_dimension_sets": len({json.dumps(r["core_dimensions"], sort_keys=True) for r in results}),
        "controls_not_already": sum(1 for c in controls if not c["positive_control"] and c["classification"] != "already_investigated"),
        "controls_implausible": sum(1 for c in controls if not c["positive_control"]),
        "positive_control_detected": all(c["classification"] in ("already_investigated", "partially_investigated")
                                         for c in controls if c["positive_control"]),
    }
    report = {
        "benchmark": "known-work novelty", "date": started.isoformat(),
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "procedure": __doc__.strip(), "metrics": metrics, "items": results, "controls": controls, "limitations": LIMITATIONS,
        "configuration": {"sources": list(checker.sources), "per_query": checker.per_query, "top_k": checker.top_k,
                          "max_queries": checker.max_queries, "citations": checker.citations,
                          "llm_judge": llm_label if checker.live_llm else None,
                          "embedding_model": os.getenv("EMBEDDING_MODEL_NAME") or "all-MiniLM-L6-v2",
                          "policy": vars(checker.policy)},
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"novelty_known_work_{started.strftime('%Y%m%d_%H%M')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["path"] = path
    return report
