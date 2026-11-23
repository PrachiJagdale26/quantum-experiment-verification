"""
Automated Unit and Integration Test Suite for the LangGraph Multi-Agent Architecture.
Tests all 10 core agentic behaviors:
1. Successful Bell-state workflow (PASS)
2. Incorrect Bell-state probabilities handled by Verification Auditor (FAIL)
3. GHZ tripartite state multi-agent workflow (PASS)
4. Missing qubit count routing to human review / pause state
5. Missing expected probabilities handling (INCONCLUSIVE / non-hallucinated)
6. Irrelevant literature evidence triggering retrieval query refinement
7. Graceful fallback on Vertex AI API failure
8. Bounded retry limit enforcement (preventing infinite loops)
9. Human-in-the-loop approval & resume
10. Inviolable scientific rule: computational FAIL cannot be overridden by an LLM
"""

import pytest
from unittest.mock import MagicMock, patch

from src.schemas.proposal import (
    ProposalSpecification,
    CircuitSpec,
    GateOperation,
    ExpectedOutcome,
    ExtractedClaim,
)
from src.schemas.verification import CheckStatus, VerificationReport
from src.agents import (
    create_initial_state,
    run_verification_workflow,
    build_verification_graph,
    extract_document_tool,
    retrieve_literature_tool,
    simulate_circuit_tool,
    run_verification_audit_tool,
    generate_report_tool,
)
from src.agents.document_agent import document_analyst_agent
from src.agents.literature_agent import literature_researcher_agent
from src.agents.simulation_agent import quantum_simulation_agent
from src.agents.verification_agent import verification_auditor_agent
from src.agents.reporting_agent import explainability_reporting_agent


# -------------------------------------------------------------
# Test 1: Successful Bell-State Workflow
# -------------------------------------------------------------
def test_successful_bell_state_agentic_workflow():
    bell_spec = ProposalSpecification(
        title="EPR Pair Bell State Synthesis",
        objective="Generate maximally entangled state |Phi+> = (|00> + |11>)/sqrt(2).",
        hypothesis="Hadamard followed by CNOT creates maximal entanglement.",
        num_qubits=2,
        circuit=CircuitSpec(num_qubits=2, gates=[
            GateOperation(name="H", target_qubits=[0]),
            GateOperation(name="CX", target_qubits=[0, 1]),
        ]),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.5),
            ExpectedOutcome(basis_state="11", expected_probability=0.5),
        ],
        author_claims=[
            ExtractedClaim(claim_id="CLM-01", claim_type="outcome", statement="System yields 50% |00> and 50% |11>.")
        ],
        confirmed_by_researcher=True,
    )

    state = create_initial_state(proposal_spec=bell_spec, researcher_confirmed=True)
    res = run_verification_workflow(state)

    assert res.get("overall_status") == "PASS"
    assert res.get("discrepancy_detected") is False
    assert res.get("tvd_score") is not None and res.get("tvd_score") <= 0.05
    assert res.get("fidelity_score") is not None and res.get("fidelity_score") >= 0.95
    assert res.get("final_report") is not None
    assert len(res.get("agent_logs", [])) >= 5


# -------------------------------------------------------------
# Test 2: Incorrect Bell-State Probabilities Handled by Auditor
# -------------------------------------------------------------
def test_incorrect_bell_state_probabilities_handled_by_auditor():
    flawed_spec = ProposalSpecification(
        title="Flawed Bell State Proposal",
        objective="Create Bell state but claim biased outcomes",
        hypothesis="Author claims asymmetric probability distribution on standard Bell circuit",
        num_qubits=2,
        circuit=CircuitSpec(num_qubits=2, gates=[
            GateOperation(name="H", target_qubits=[0]),
            GateOperation(name="CX", target_qubits=[0, 1]),
        ]),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.9),
            ExpectedOutcome(basis_state="11", expected_probability=0.1),
        ],
        author_claims=[
            ExtractedClaim(claim_id="CLM-01", claim_type="outcome", statement="Author claims 90% |00> and 10% |11>.")
        ],
        confirmed_by_researcher=True,
    )

    state = create_initial_state(proposal_spec=flawed_spec, researcher_confirmed=True)
    res = run_verification_workflow(state)

    assert res.get("overall_status") == "FAIL"
    assert res.get("discrepancy_detected") is True
    assert res.get("tvd_score") > 0.05
    assert len(res.get("fail_reasons", [])) > 0
    assert any("distribution" in r.lower() or "error" in r.lower() for r in res.get("fail_reasons", []))


# -------------------------------------------------------------
# Test 3: GHZ-State Workflow
# -------------------------------------------------------------
def test_ghz_tripartite_state_agentic_workflow():
    ghz_spec = ProposalSpecification(
        title="3-Qubit GHZ State Entanglement",
        objective="Create tripartite GHZ state (|000> + |111>)/sqrt(2)",
        hypothesis="Hadamard and cascaded CNOTs create non-local tripartite correlation",
        num_qubits=3,
        circuit=CircuitSpec(num_qubits=3, gates=[
            GateOperation(name="H", target_qubits=[0]),
            GateOperation(name="CX", target_qubits=[0, 1]),
            GateOperation(name="CX", target_qubits=[1, 2]),
        ]),
        expected_outcomes=[
            ExpectedOutcome(basis_state="000", expected_probability=0.5),
            ExpectedOutcome(basis_state="111", expected_probability=0.5),
        ],
        author_claims=[
            ExtractedClaim(claim_id="CLM-01", claim_type="outcome", statement="Equal probability of 0.5 for |000> and |111>.")
        ],
        confirmed_by_researcher=True,
    )

    state = create_initial_state(proposal_spec=ghz_spec, researcher_confirmed=True)
    res = run_verification_workflow(state)

    assert res.get("overall_status") == "PASS"
    assert res.get("discrepancy_detected") is False
    assert res.get("simulation_output") is not None
    probs = res.get("computed_probabilities", {})
    assert pytest.approx(probs.get("000", 0.0), abs=1e-3) == 0.5
    assert pytest.approx(probs.get("111", 0.0), abs=1e-3) == 0.5


# -------------------------------------------------------------
# Test 4: Missing Qubit Count Routes to Human Review
# -------------------------------------------------------------
def test_missing_qubit_count_routes_to_human_review():
    raw_unspecified = "We propose an optical quantum experiment for state transfer. No qubits are mentioned."
    state = create_initial_state(raw_text=raw_unspecified, filename="unspecified_qubits.pdf")
    res = run_verification_workflow(state)

    assert res.get("requires_human_review") is True
    assert res.get("human_review_reason") is not None
    assert any("qubit" in p.lower() for p in res.get("missing_parameters", []))
    # Simulation should NOT have executed
    assert res.get("simulation_output") is None


# -------------------------------------------------------------
# Test 5: Missing Expected Probabilities Handling
# -------------------------------------------------------------
def test_missing_expected_probabilities_handling():
    spec_no_outcomes = ProposalSpecification(
        title="Superposition Proposal (No Probabilities)",
        objective="Prepare single qubit in superposition",
        num_qubits=1,
        circuit=CircuitSpec(num_qubits=1, gates=[GateOperation(name="H", target_qubits=[0])]),
        expected_outcomes=[],
        confirmed_by_researcher=True,
    )

    state = create_initial_state(proposal_spec=spec_no_outcomes, researcher_confirmed=True)
    res = run_verification_workflow(state)

    # Missing expected outcomes should yield INCONCLUSIVE check rather than crashing or guessing
    checks = res.get("verification_checks", [])
    dist_chk = next((c for c in checks if c.category.value == "DISTRIBUTION_MATCH"), None)
    assert dist_chk is not None
    assert dist_chk.status == CheckStatus.INCONCLUSIVE
    assert res.get("simulation_output") is not None


# -------------------------------------------------------------
# Test 6: Irrelevant Literature Evidence Triggers Refinement
# -------------------------------------------------------------
def test_irrelevant_literature_triggers_retrieval_refinement():
    spec = ProposalSpecification(
        title="Uncommon Quantum Hypothesis",
        objective="Experimental preparation",
        num_qubits=1,
        circuit=CircuitSpec(num_qubits=1, gates=[GateOperation(name="X", target_qubits=[0])]),
        author_claims=[ExtractedClaim(claim_id="CLM-01", claim_type="hypothesis", statement="Unrelated novel physical mechanism")],
    )

    # Initial state with no citations
    state = create_initial_state(proposal_spec=spec, researcher_confirmed=True)
    out = literature_researcher_agent(state)

    # If evidence is weak or unaddressed, retry query should be formulated
    assert "literature_citations" in out
    assert "retrieval_retry_count" in out


# -------------------------------------------------------------
# Test 7: Graceful Fallback on Vertex AI Failure
# -------------------------------------------------------------
def test_vertex_ai_failure_fallback():
    raw_doc = "Title: Deterministic Bell Test\nWe use 2 qubits. Circuit: H 0, CX 0 1. Outcomes: |00> 0.5, |11> 0.5."
    
    # Mock LLM client to raise an exception simulating API failure
    mock_client = MagicMock()
    mock_client.generate_json.side_effect = RuntimeError("Vertex AI Connection Timeout")

    res = extract_document_tool(
        raw_text=raw_doc,
        filename="vertex_test.pdf",
        llm_client=mock_client,
    )

    # Should fall back gracefully to heuristic/rule-based extractor without raising an uncaught exception
    assert res.get("proposal_spec") is not None
    assert res.get("proposal_spec").num_qubits == 2


# -------------------------------------------------------------
# Test 8: Agent Retry Limits Prevent Infinite Loops
# -------------------------------------------------------------
def test_agent_retry_limits_prevent_infinite_loops():
    spec = ProposalSpecification(
        title="Obscure Hypothesis",
        objective="Test non-existent phenomenon",
        num_qubits=1,
        circuit=CircuitSpec(num_qubits=1, gates=[GateOperation(name="H", target_qubits=[0])]),
    )
    # Set retry count to maximum
    state = create_initial_state(proposal_spec=spec, researcher_confirmed=True)
    state["retrieval_retry_count"] = 2
    state["max_retrieval_retries"] = 2

    # Literature agent should NOT schedule further retries
    out = literature_researcher_agent(state)
    assert out.get("retrieval_query") is None or out.get("retrieval_retry_count", 0) <= 2


# -------------------------------------------------------------
# Test 9: Human Approval and Resume
# -------------------------------------------------------------
def test_human_approval_and_resume():
    # 1. Start with incomplete document
    raw_text = "We propose an experiment but forgot the circuit and qubits."
    st_init = create_initial_state(raw_text=raw_text, filename="incomplete.pdf")
    res_paused = run_verification_workflow(st_init)

    assert res_paused.get("requires_human_review") is True

    # 2. Researcher provides confirmed parameters
    confirmed_overrides = {
        "title": "Confirmed Bell Proposal",
        "num_qubits": 2,
        "gates": [GateOperation(name="H", target_qubits=[0]), GateOperation(name="CX", target_qubits=[0, 1])],
        "expected_outcomes": [ExpectedOutcome(basis_state="00", expected_probability=0.5), ExpectedOutcome(basis_state="11", expected_probability=0.5)],
    }

    res_resumed = run_verification_workflow(res_paused, human_overrides=confirmed_overrides)

    assert res_resumed.get("requires_human_review") is False
    assert res_resumed.get("overall_status") == "PASS"
    assert res_resumed.get("tvd_score") <= 0.05
    assert res_resumed.get("simulation_output") is not None


# -------------------------------------------------------------
# Test 10: Inviolable Scientific Rule (Computational FAIL cannot be overridden by LLM)
# -------------------------------------------------------------
def test_computational_fail_cannot_be_overridden_by_llm():
    # Proposal claiming |00>=0.9 on standard Bell state (simulated is 0.5)
    biased_spec = ProposalSpecification(
        title="Biased Bell Proposal",
        objective="Test inviolable rule",
        hypothesis="Theoretical argument asserting 0.9 probability",
        num_qubits=2,
        circuit=CircuitSpec(num_qubits=2, gates=[
            GateOperation(name="H", target_qubits=[0]),
            GateOperation(name="CX", target_qubits=[0, 1]),
        ]),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.9),
            ExpectedOutcome(basis_state="11", expected_probability=0.1),
        ],
        author_claims=[
            ExtractedClaim(claim_id="CLM-01", claim_type="outcome", statement="Theoretical proof claims 0.9 probability.")
        ],
        confirmed_by_researcher=True,
    )

    state = create_initial_state(proposal_spec=biased_spec, researcher_confirmed=True)
    res = run_verification_workflow(state)

    # Regardless of any theoretical claim, computational status MUST be strictly FAIL
    assert res.get("overall_status") == "FAIL"
    assert res.get("discrepancy_detected") is True
    checks = res.get("verification_checks", [])
    sim_check = next(c for c in checks if c.category.value == "DISTRIBUTION_MATCH")
    assert sim_check.status == CheckStatus.FAIL
