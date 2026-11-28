"""
Independent Evaluation Runner for Quantum Experiment Verification System.
Executes the reproducible benchmark suite, calculates metrics, and exports CSV.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.evaluation.evaluator import VerificationEvaluator


def run_gaps():
    """Gap-finding + planning benchmark on the synthetic sample papers (offline, demo mode)."""
    from src.evaluation.gap_benchmark import run_gap_benchmark
    from src.knowledge_base.retriever import QuantumRetriever
    from src.knowledge_base.vector_store import QuantumKnowledgeBase
    from src.llm.client import ConfigurableLLMClient
    from src.llm.extractor import ProposalExtractor
    from src.research import ExperimentPlanner, GapFinder, PaperLibrary, ResearchPipeline
    from src.verification.engine import VerificationEngine

    llm = ConfigurableLLMClient(demo_mode=True)
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    library = PaperLibrary(ProposalExtractor(llm), kb)
    library.load_samples()
    pipe = ResearchPipeline(library, GapFinder(llm, kb), ExperimentPlanner(llm),
                            VerificationEngine(retriever=QuantumRetriever(kb), llm_client=llm))
    m = run_gap_benchmark(pipe)

    print("=" * 80)
    print("RESEARCH-GAP IDENTIFICATION & EXPERIMENT PLANNING BENCHMARK (synthetic papers)")
    print("=" * 80)
    print(f"Papers analysed            : {m['papers']}")
    print(f"Gaps found                 : {m['gaps_found']}")
    print(f"Planted-gap recall         : {m['planted_recall']:.2f}  ({m['planted_gaps']} planted)")
    print(f"Precision (lower bound)    : {m['precision_lower_bound']:.2f}")
    print(f"Gaps with quoted evidence  : {m['evidence_coverage']:.2f}")
    print(f"Plans verified first time  : {m['plans_first_pass_rate']:.2f}")
    print(f"Plans verified after 1 fix : {m['plans_pass_after_revision_rate']:.2f}")
    print("\nPlanted gap -> matching found gaps")
    for gid, ids in m["matches"].items():
        print(f"  {gid}: {', '.join(ids) if ids else 'NOT FOUND'}")
    print("\nTop gaps")
    for g in pipe.gaps[:5]:
        print(f"  {g.gap_id} [{g.gap_type}] priority {g.priority:.2f}  {g.title}")


def run_novelty():
    """
    Known-work novelty benchmark (Phase 4), live: OpenAlex, arXiv and Crossref are queried (responses cached).
    Runs on a COPY of the research database so benchmark gaps and searches never touch your corpus.
        python evaluate.py --novelty [--n 10] [--db sqlite:///path.db] [--judge]
    """
    import sqlite3
    arg = lambda flag, default=None: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    root = os.path.abspath(os.path.dirname(__file__))
    src = arg("--db") or os.getenv("RESEARCH_DB_URL") or f"sqlite:///{os.path.join(root, 'data', 'research.db')}"
    copy = os.path.join(root, "data", "benchmark", "novelty_bench.db")
    os.makedirs(os.path.dirname(copy), exist_ok=True)
    if os.path.exists(copy):
        os.remove(copy)
    a, b = sqlite3.connect(src.replace("sqlite:///", "")), sqlite3.connect(copy)
    a.backup(b)                       # the backup API keeps WAL-mode databases consistent
    a.close(), b.close()
    db_url = f"sqlite:///{copy}"
    os.environ["RESEARCH_DB_URL"] = db_url

    from src.evaluation.novelty_benchmark import run_known_work
    from src.gap_discovery import NoveltyChecker
    llm, label = None, None
    if "--judge" in sys.argv:
        from src.llm.client import ConfigurableLLMClient
        llm = ConfigurableLLMClient(demo_mode=False)
        label = f"{llm.provider_name}/{llm.model_name}"
    report = run_known_work(db_url, n=int(arg("--n", 10)), checker=NoveltyChecker(db_url, llm=llm),
                            out_dir=os.path.join(root, "data", "benchmark"), llm_label=label)
    m = report["metrics"]
    print("=" * 80)
    print("KNOWN-WORK NOVELTY BENCHMARK (live search; ground truth: the question is published)")
    print("=" * 80)
    print(f"Items                      : {m['items']}  (from {src})")
    print(f"False-novelty rate         : {m['false_novelty_rate']}  95% CI {m['false_novelty_ci95']}")
    print(f"Detected (already/partial) : {m['detection_rate']}")
    print(f"Insufficient coverage      : {m['insufficient_coverage_rate']}")
    print(f"Source paper screened      : {m['recall_screened']}")
    print(f"Source paper compared      : {m['recall_compared_top_k']}   MRR {m['mrr_compared']}")
    print(f"By classification          : {m['by_classification']}")
    print(f"Distinct question shapes   : {m['distinct_core_dimension_sets']} of {m['items']}")
    print(f"Specificity control        : {m['controls_not_already']} of {m['controls_implausible']} implausible combinations "
          f"not 'already investigated'; positive control detected: {m['positive_control_detected']}")
    print(f"Model judge                : {label or 'none (rule-based comparison only)'}")
    print(f"Duration                   : {report['duration_s']} s")
    print(f"Full report                : {report.get('path')}")
    for r in report["items"]:
        print(f"  [{r['classification']:<24}] rank={r['rank_in_compared']} {r['title'][:70]}")


def run_all_evaluations():
    """
    Every evaluation, on a copy of the research database; writes data/benchmark/evaluation_*.json and docs/EVALUATION.md.
        python evaluate.py --all [--db sqlite:///path.db]
    """
    arg = lambda flag, default=None: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    root = os.path.abspath(os.path.dirname(__file__))
    src = arg("--db") or os.getenv("RESEARCH_DB_URL") or f"sqlite:///{os.path.join(root, 'data', 'research.db')}"
    from src.evaluation.run_all import run_all, to_markdown
    from src.research_analysis.cross_paper_analysis import default_embedder
    res = run_all(src, embed=default_embedder())
    md = to_markdown(res)
    with open(os.path.join(root, "docs", "EVALUATION.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"Full results: {res['path']}")
    print("Labelling sheets: " + ", ".join(f"{k} ({v})" for k, v in res["labelling_sheets"].items()))


def main():
    if "--all" in sys.argv:
        run_all_evaluations()
        return
    if "--novelty" in sys.argv:
        run_novelty()
        return
    if "--gaps" in sys.argv:
        run_gaps()
        return
    print("=" * 80)
    print("AI-BASED VERIFICATION OF QUANTUM EXPERIMENT PROPOSALS - BENCHMARK EVALUATION")
    print("=" * 80)

    csv_path = "evaluation_results.csv"
    evaluator = VerificationEvaluator(output_csv=csv_path)

    print(f"Loaded {len(evaluator.cases)} labelled benchmark test cases.")
    print("Executing multi-tier evaluation...\n")

    df, summary = evaluator.run_evaluation()

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION RESULTS (AGGREGATED BY MODE)")
    print("=" * 80)
    
    header = f"{'Verification Mode':<30} | {'Prec.':<7} | {'Recall':<7} | {'FPR':<7} | {'FAR':<7} | {'Sim Agree':<10} | {'Latency (ms)':<12}"
    print(header)
    print("-" * len(header))

    for mode, m in summary.items():
        print(
            f"{mode:<30} | "
            f"{m['precision']:<7.4f} | "
            f"{m['recall']:<7.4f} | "
            f"{m['false_pass_rate']:<7.4f} | "
            f"{m['false_alarm_rate']:<7.4f} | "
            f"{m['simulation_agreement']:<10.4f} | "
            f"{m['avg_execution_time_ms']:<12.2f}"
        )

    print("\n" + "=" * 80)
    print("PER-CASE BREAKDOWN (HYBRID MODE)")
    print("=" * 80)
    hybrid_df = df[df["mode"] == "Hybrid (Rules + Sim + RAG)"]
    for _, row in hybrid_df.iterrows():
        status_flag = "PASS" if row["is_correct"] else "FAIL"
        print(
            f"[{row['case_id']}] {row['case_name'][:45]:<45} | "
            f"Expected: {row['expected_status']:<12} | "
            f"Actual: {row['actual_status']:<12} | "
            f"Agreement: {status_flag}"
        )

    print(f"\nDetailed evaluation metrics saved to: {os.path.abspath(csv_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
