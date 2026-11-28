"""
Runs every evaluation and writes the results (JSON) and docs/EVALUATION.md.

Everything here runs on a COPY of the research database (made with SQLite's backup API), so evaluation never
changes the researcher's data. Retrieval and novelty are read from the stored known-work benchmark
(`python evaluate.py --novelty` runs it live); everything else is computed now, offline.
"""

import glob
import json
import os
import platform
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from sqlalchemy import select

from ..storage import ExtractionRecord, session_scope
from .extraction_eval import corpus_statistics, verifier_stress_test
from .gap_eval import evaluate_gaps
from .labels import create_sheets, score_sheets
from .retrieval_eval import evaluate_retrieval
from .verification_eval import evaluate_verification

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _copy(src_url: str, dst_path: str) -> str:
    if os.path.exists(dst_path):
        os.remove(dst_path)
    a, b = sqlite3.connect(src_url.replace("sqlite:///", "")), sqlite3.connect(dst_path)
    a.backup(b)
    a.close(), b.close()
    return f"sqlite:///{dst_path}"


def _versions() -> Dict[str, str]:
    import importlib.metadata as m
    out = {"python": platform.python_version()}
    for pkg in ("qiskit", "qiskit-aer", "langgraph", "sentence-transformers", "chromadb", "rapidfuzz"):
        try:
            out[pkg] = m.version(pkg)
        except Exception:
            out[pkg] = "not installed"
    out["embedding_model"] = os.getenv("EMBEDDING_MODEL_NAME") or "all-MiniLM-L6-v2"
    return out


def run_all(src_db: str, embed: Optional[Callable] = None, out_dir: Optional[str] = None,
            labels_dir: Optional[str] = None) -> Dict[str, Any]:
    out_dir = out_dir or os.path.join(ROOT, "data", "benchmark")
    labels_dir = labels_dir or os.path.join(out_dir, "labels")
    os.makedirs(out_dir, exist_ok=True)
    db = _copy(src_db, os.path.join(out_dir, "eval_copy.db"))
    started = datetime.now(timezone.utc)
    with session_scope(db) as s:
        models = Counter(r.extractor for r in s.scalars(select(ExtractionRecord)) if not r.field.startswith("dim:"))
    res: Dict[str, Any] = {"date": started.isoformat(timespec="seconds"), "database": src_db.split("/")[-1],
                           "software": _versions(), "extractors": dict(models)}
    res["extraction"] = {"corpus": corpus_statistics(db), "verifier_stress_test": verifier_stress_test(db)}
    res["gaps"] = evaluate_gaps(db, embed=embed)
    res["verification"] = evaluate_verification()
    bench_db = os.path.join(out_dir, "novelty_bench.db")
    res["retrieval"] = evaluate_retrieval(f"sqlite:///{bench_db}") if os.path.exists(bench_db) else {
        "not_measured": "Run `python evaluate.py --novelty` first; retrieval is read from its stored searches."}
    reports = sorted(glob.glob(os.path.join(out_dir, "novelty_known_work_*.json")))
    if reports:
        nk = json.load(open(reports[-1], encoding="utf-8"))
        res["novelty"] = {"source": os.path.basename(reports[-1]), "date": nk["date"], "metrics": nk["metrics"],
                          "controls": nk.get("controls", []), "configuration": nk["configuration"],
                          "limitations": nk["limitations"]}
    else:
        res["novelty"] = {"not_measured": "Run `python evaluate.py --novelty`."}
    res["labelling_sheets"] = create_sheets(db, labels_dir)
    res["human_labelled"] = score_sheets(labels_dir)
    res["duration_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    path = os.path.join(out_dir, f"evaluation_{started.strftime('%Y%m%d_%H%M')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    res["path"] = path
    from ..storage.db import _engines
    eng = _engines.pop(db, None)                      # release the copy so it can be deleted (Windows locks open files)
    if eng is not None:
        eng.dispose()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(os.path.join(out_dir, "eval_copy.db" + suffix))
        except OSError:
            pass
    return res


def _fmt(v):
    return "not measured" if v is None else v


def to_markdown(r: Dict[str, Any]) -> str:
    ex, ver, gp = r["extraction"], r["verification"], r["gaps"]
    st, cs = ex["verifier_stress_test"], ex["corpus"]
    L = [f"# Evaluation results", "",
         f"Measured {r['date'][:10]} on a copy of the research database (file `{r['database']}`; {cs['papers']} analysed papers, "
         f"{cs['from_full_text']} from full text). Software: " + ", ".join(f"{k} {v}" for k, v in r["software"].items()) + ".",
         "Every number below was measured by `python evaluate.py --all` (novelty and retrieval come from the stored "
         "`python evaluate.py --novelty` run). Nothing is estimated; metrics that need human labels say so.", ""]
    L += ["## 1. Novelty assessment (known-work set)", ""]
    nv = r["novelty"]
    if "metrics" in nv:
        m = nv["metrics"]
        L += [f"Procedure: each analysed paper's own verified question became a candidate gap; the paper was removed from "
              f"the local corpus and the full novelty check ran against OpenAlex, arXiv and Crossref ({nv['date'][:10]}, "
              f"model judge: {nv['configuration'].get('llm_judge') or 'none'}).", "",
              "| Metric | Value |", "|---|---|",
              f"| Items | {m['items']} |", f"| False-novelty rate | {m['false_novelty_rate']} (95% CI {m['false_novelty_ci95']}) |",
              f"| Detected as already / partially investigated | {m['detection_rate']} |",
              f"| Insufficient coverage | {m['insufficient_coverage_rate']} |",
              f"| Source paper among screened / compared | {m['recall_screened']} / {m['recall_compared_top_k']} |",
              f"| Distinct question shapes | {m['distinct_core_dimension_sets']} of {m['items']} |",
              f"| Specificity control: implausible combinations not called “already investigated” | "
              f"{m['controls_not_already']} of {m['controls_implausible']} |",
              f"| Positive control detected | {m['positive_control_detected']} |", "",
              "Limitations: " + " ".join(nv["limitations"]), ""]
    else:
        L += [nv["not_measured"], ""]
    L += ["## 2. Literature retrieval (known-work set)", ""]
    rt = r["retrieval"]
    if "any_query_any_source" in rt:
        L += [f"Procedure: rank of the known source paper in the stored results of every query ({rt['gaps']} gaps; "
              f"{rt['failed_or_rate_limited_searches']} failed searches). Up to 10 results per query, so Recall@10 is the ceiling.", "",
              "| Pool | n | Recall@1 | Recall@3 | Recall@10 | MRR |", "|---|---|---|---|---|---|"]
        rows = [("All queries, all sources", rt["any_query_any_source"])] + \
               [(f"Source: {k}", v) for k, v in rt["by_source"].items()] + [(f"Query: {k}", v) for k, v in rt["by_query_kind"].items()]
        L += [f"| {n} | {v['n']} | {v['recall@1']} | {v['recall@3']} | {v['recall@10']} | {v['mrr']} |" for n, v in rows]
        L += ["", "Limitations: one known relevant paper per query, so precision is not measured; the gap text is the "
                  "paper's own sentence, which favours the own-words query.", ""]
    else:
        L += [rt["not_measured"], ""]
    L += ["## 3. Extraction", "",
          f"Corpus: {cs['statements']} statements from {cs['papers']} papers ({cs['by_kind']}); "
          f"{cs['model_evidence_verified_share']} of model statements had their quoted evidence found in the source.", "",
          f"Verifier stress test on {st['quotes']} verified quotes from {st['papers']} papers: detection of fabricated quotes "
          f"**{st['detection_rate']}**, false rejection of genuine quotes **{st['false_rejection_rate']}**.", "",
          "| Variant | Accepted / total |", "|---|---|"]
    L += [f"| {k} | {v['accepted']} / {v['total']} |" for k, v in st["by_variant"].items()]
    L += ["", "Phase 9 found that fuzzy matching alone accepted every changed number and inserted negation and 73% of "
              "swapped terms (detection 0.52). The verifier now also requires numbers, negations and content words to "
              "agree with the matched passage; the table shows the result after that change.", ""]
    hl = r["human_labelled"]
    L += [f"Field accuracy: {_fmt(hl['extraction_field_accuracy']['value'])} ({hl['extraction_field_accuracy']['labelled']} of "
          f"{hl['extraction_field_accuracy']['rows']} rows labelled). Attribution accuracy: "
          f"{_fmt(hl['extraction_attribution_accuracy']['value'])} ({hl['extraction_attribution_accuracy']['labelled']} labelled).", ""]
    L += ["## 4. Gaps", "",
          f"{gp.get('gaps', 0)} candidate gaps ({gp.get('by_origin')}). Supporting evidence is a verified quote for "
          f"{gp.get('evidence_quoted_share')}; {gp.get('specific_share')} name two or more research dimensions of their own; "
          f"{len(gp.get('near_duplicate_pairs', []))} near-duplicate pair(s) at cosine ≥ 0.85; novelty outcomes "
          f"{gp.get('novelty_outcomes')}.", "",
          f"Expert ratings: relevance {_fmt(hl['gap_relevance_mean']['value'])}, evidence support "
          f"{_fmt(hl['gap_evidence_support_mean']['value'])} ({hl['gap_relevance_mean']['labelled']} of "
          f"{hl['gap_relevance_mean']['rows']} labelled).", ""]
    L += ["## 5. Experiment plans", "",
          "Rubric means (1-5): " + ", ".join(f"{k} {_fmt(v['value'])} ({v['labelled']} of {v['rows']} labelled)"
                                             for k, v in hl["plan_rubric_means"].items()) + ".", ""]
    L += ["## 6. Quantum verification", "", "| Set | Cases | Correct | False pass | False fail |", "|---|---|---|---|---|"]
    for name, key in (("Proposal benchmark", "proposal_benchmark"), ("Plan cases (textbook ground truth)", "plan_cases")):
        v = ver[key]
        L.append(f"| {name} | {v['cases']} | {v['correct']} | {v['false_pass']} of {v['of_should_not_pass']} | "
                 f"{v['false_fail']} of {v['of_should_pass']} |")
    L += ["", "Plan cases: " + "; ".join(f"{c['id']} → {c['got']}" for c in ver["plan_cases"]["cases_detail"]) + ".", ""]
    L += ["## 7. Future-work and partially-investigated sets", "",
          f"Future-work statements addressed later: {_fmt(hl['future_work_addressed_share']['value'])} "
          f"({hl['future_work_addressed_share']['labelled']} of {hl['future_work_addressed_share']['rows']} labelled). "
          f"Partially-investigated variants labelled: {hl['partially_investigated_labelled']['labelled']} of "
          f"{hl['partially_investigated_labelled']['rows']}.", "",
          "## How to add the human-labelled numbers", "",
          "Fill the `label_*` columns in `data/benchmark/labels/*.csv` and run `python evaluate.py --all` again. "
          "Existing sheets are never overwritten.", "",
          "## Limitations", "",
          "- A small corpus on one topic (entanglement and decoherence); rates have wide uncertainty.",
          "- Most papers were analysed from abstracts only.",
          "- Novelty and retrieval ground truth is one known published paper per question.",
          "- Verification ground truth covers small textbook circuits and the 6 hand-written benchmark cases.",
          "- Human-rated metrics are not measured until labelled.", ""]
    return "\n".join(L)
