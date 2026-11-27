"""Stage 3: experiment planning from research gaps (offline template planner)."""

import pytest

from src.circuit.simulator import QuantumSimulator
from src.research.domain import template_circuit
from src.schemas.research_gap import GapEvidence, ResearchGap
from tests.research_fixtures import build_pipeline


@pytest.fixture(scope="module")
def pipe():
    p = build_pipeline()
    p.find_gaps()
    return p


def test_every_gap_gets_a_plan_with_executable_circuit(pipe):
    sim = QuantumSimulator()
    papers = [p for _, p in pipe.library.papers()]
    for gap in pipe.gaps:
        plan = pipe.planner.plan(gap, papers, f"T-{gap.gap_id}")
        assert plan.hypothesis and plan.method_steps and plan.success_metrics
        circ = plan.proposal.circuit
        assert circ is not None and 1 <= circ.num_qubits <= 10
        out = sim.simulate(circ)
        assert out.is_normalized
        total = sum(o.expected_probability for o in plan.proposal.expected_outcomes)
        assert abs(total - 1.0) < 1e-6


@pytest.mark.parametrize("gtype,hints,expected_n", [
    ("scale_limit", {"family": "ghz", "max_qubits": "3"}, 4),
    ("unexplored_combination", {"family": "bell", "target_system": "Trapped Ion"}, 2),
    ("conflicting_results", {"family": "bell", "other_paper": "Paper D"}, 2),
    ("future_work", {"family": "vqe"}, 2),
    ("stated_limitation", {}, 2),
])
def test_template_per_gap_type(pipe, gtype, hints, expected_n):
    gap = ResearchGap(gap_id="GAP-X", title="Test gap", statement="Readout mitigation was not applied.",
                      gap_type=gtype, evidence=[GapEvidence(paper_title="P", quote="q")], hints=hints)
    plan = pipe.planner.plan(gap, [], "T-1")
    assert plan.proposal.circuit.num_qubits == expected_n
    assert plan.generated_by == "template"
    if gtype == "unexplored_combination":
        assert "trapped ion" in plan.title.lower()


def test_template_predictions_come_from_theory():
    n, gates, outcomes, why = template_circuit("ghz", 5)
    assert n == 5 and len(gates) == 5
    assert {o.basis_state for o in outcomes} == {"00000", "11111"}
    assert "GHZ" in why
