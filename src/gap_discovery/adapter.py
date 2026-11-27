"""Bridge from a corpus gap to the existing planner's ResearchGap (Phase 5 replaces the planner itself)."""

from typing import Dict

from ..schemas.research_gap import GapEvidence, ResearchGap
from .gap_ranker import SIMULATOR_QUBITS
from .schema import Gap

GAP_TYPE = {"stated_limitation": "stated_limitation", "unresolved_problem": "future_work", "future_work": "future_work",
            "unexplored_combination": "unexplored_combination"}


def to_research_gap(gap: Gap, scores: Dict[str, dict], total: float) -> ResearchGap:
    qubits = [int(v) for v in (gap.dimensions["qubit_scale"].values if "qubit_scale" in gap.dimensions else []) if str(v).isdigit()]
    hints = {"source": "corpus", "origin": gap.origin}
    if "platform" in gap.dimensions:
        hints["target_system"] = gap.dimensions["platform"].values[0]
    if qubits:
        hints["qubits"] = str(min(qubits))
    sup = [e for e in gap.evidence if e.role == "supporting"]
    if sup:
        hints["source_paper"] = sup[0].title
    return ResearchGap(
        gap_id=f"CG-{gap.id}", title=gap.title, statement=gap.research_question or gap.statement,
        gap_type=GAP_TYPE.get(gap.origin, "stated_limitation"),
        evidence=[GapEvidence(paper_title=e.title, page_number=e.page, quote=e.quote) for e in gap.evidence],
        novelty=scores["novelty"]["score"], feasibility=scores["feasibility"]["score"],
        evidence_strength=scores["evidence"]["score"], priority=total,
        testable_on_simulator=not qubits or max(qubits) <= SIMULATOR_QUBITS, hints=hints)
