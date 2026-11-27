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


def main():
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
