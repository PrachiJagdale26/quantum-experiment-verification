"""
Tests for the distribution fidelity shown in reports and the dashboard header.
"""

import pytest

from src.circuit.metrics import distribution_fidelity
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.verification.engine import VerificationEngine

BELL = {"00": 0.5, "11": 0.5}


def test_identical_distributions_have_fidelity_one():
    assert distribution_fidelity(BELL, BELL) == pytest.approx(1.0)


def test_biased_bell_expectation():
    # (sqrt(0.9*0.5) + sqrt(0.1*0.5))^2 = 0.8
    assert distribution_fidelity({"00": 0.9, "11": 0.1}, BELL) == pytest.approx(0.8)


def test_disjoint_distributions_have_fidelity_zero():
    assert distribution_fidelity({"01": 1.0}, BELL) == pytest.approx(0.0)


def test_unmentioned_simulated_probability_counts_against_proposal():
    # Proposal only claims |00>; half the simulated probability is elsewhere.
    assert distribution_fidelity({"00": 1.0}, BELL) == pytest.approx(0.5)


def test_empty_distribution_returns_none():
    assert distribution_fidelity({}, BELL) is None


def _bell_proposal(p00, p11):
    return ProposalSpecification(
        title="Bell",
        objective="Prepare a Bell state.",
        num_qubits=2,
        circuit=CircuitSpec(num_qubits=2, gates=[
            GateOperation(name="H", target_qubits=[0]),
            GateOperation(name="CX", target_qubits=[0, 1]),
        ]),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=p00),
            ExpectedOutcome(basis_state="11", expected_probability=p11),
        ],
    )


def test_engine_reports_fidelity():
    engine = VerificationEngine(retriever=None, llm_client=None)
    assert engine.verify_proposal(_bell_proposal(0.5, 0.5)).fidelity_score == pytest.approx(1.0)
    assert engine.verify_proposal(_bell_proposal(0.9, 0.1)).fidelity_score == pytest.approx(0.8)
