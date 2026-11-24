"""
Tests for the re-extraction review node (Workflow C) and the literature retry strategy.
"""

from src.agents.orchestrator import re_extraction_review_node, route_after_literature
from src.agents.literature_agent import build_refined_query
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome


def _bell_spec(num_qubits=2, gates=None):
    gates = gates or [
        GateOperation(name="H", target_qubits=[0]),
        GateOperation(name="CX", target_qubits=[0, 1]),
    ]
    return ProposalSpecification(
        title="Bell State Preparation",
        objective="Prepare a Bell state.",
        hypothesis="H then CNOT yields equal 00/11 outcomes.",
        num_qubits=num_qubits,
        circuit=CircuitSpec(num_qubits=num_qubits, gates=gates),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.8),
            ExpectedOutcome(basis_state="11", expected_probability=0.2),
        ],
    )


BELL_TEXT = (
    "We prepare a two-qubit circuit. Apply H on qubit 0, then CX on qubits 0, 1. "
    "We expect P(00)=0.8 and P(11)=0.2."
)


def _details(out):
    return out["agent_logs"][0]["details"]


def test_re_extraction_confirms_matching_spec():
    out = re_extraction_review_node({
        "document_raw_text": BELL_TEXT,
        "proposal_spec": _bell_spec(),
        "fail_reasons": ["distribution mismatch"],
    })
    assert out["re_extraction_attempted"] is True
    assert out["re_extraction_outcome"] == "CONFIRMED"
    assert _details(out)["mismatches"] == []
    assert "potential_inconsistencies" not in out


def test_re_extraction_flags_spec_that_disagrees_with_source():
    # Spec claims 3 qubits / different gates than the source text describes.
    wrong = _bell_spec(num_qubits=3, gates=[GateOperation(name="X", target_qubits=[2])])
    out = re_extraction_review_node({
        "document_raw_text": BELL_TEXT,
        "proposal_spec": wrong,
        "fail_reasons": ["distribution mismatch"],
        "potential_inconsistencies": [],
    })
    assert out["re_extraction_outcome"] == "POSSIBLE_EXTRACTION_ARTIFACT"
    assert len(_details(out)["mismatches"]) >= 1
    assert out["potential_inconsistencies"]
    # The node must never change the verdict itself.
    assert "overall_status" not in out


def test_re_extraction_undetermined_for_vague_text():
    out = re_extraction_review_node({
        "document_raw_text": "This proposal studies entanglement in general terms.",
        "proposal_spec": _bell_spec(),
        "fail_reasons": ["distribution mismatch"],
    })
    assert out["re_extraction_outcome"] == "UNDETERMINED"


def test_refined_queries_differ_between_retries():
    spec = _bell_spec()
    q1 = build_refined_query(spec, attempt=1)
    q2 = build_refined_query(spec, attempt=2)
    assert q1 != q2
    assert "CX" in q1 and "H" in q1
    assert "|00>" in q2
    # deterministic for the same input
    assert q1 == build_refined_query(spec, attempt=1)


def test_router_follows_query_and_respects_bound():
    assert route_after_literature({"retrieval_query": "x", "retrieval_retry_count": 1, "max_retrieval_retries": 2}) == "literature_researcher"
    assert route_after_literature({"retrieval_query": None, "retrieval_retry_count": 1, "max_retrieval_retries": 2}) == "quantum_simulation"
    assert route_after_literature({"retrieval_query": "x", "retrieval_retry_count": 3, "max_retrieval_retries": 2}) == "quantum_simulation"
