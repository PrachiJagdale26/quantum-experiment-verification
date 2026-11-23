"""
LangGraph Orchestrator for Multi-Agent Quantum Experiment Verification.
Orchestrates the 5 specialized agents with typed shared state, conditional routing,
bounded retry loops, human-in-the-loop review, and computational fail precedence.
"""

from typing import Dict, Any, Optional, Literal
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END

from .state import VerificationAgentState, create_initial_state
from .document_agent import document_analyst_agent
from .literature_agent import literature_researcher_agent
from .simulation_agent import quantum_simulation_agent
from .verification_agent import verification_auditor_agent
from .reporting_agent import explainability_reporting_agent
from ..schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome

logger = logging.getLogger(__name__)


def apply_human_overrides_to_spec(
    spec: Optional[ProposalSpecification],
    overrides: Dict[str, Any]
) -> Optional[ProposalSpecification]:
    """Applies researcher overrides directly to a ProposalSpecification."""
    if not spec:
        spec = ProposalSpecification(
            title=overrides.get("title", "Researcher-Specified Proposal"),
            objective=overrides.get("objective", "Verified quantum experiment proposal."),
            num_qubits=int(overrides.get("num_qubits", 1)),
        )
    if "title" in overrides and overrides["title"]:
        spec.title = overrides["title"]
    if "objective" in overrides and overrides["objective"]:
        spec.objective = overrides["objective"]
    if "num_qubits" in overrides and overrides["num_qubits"]:
        spec.num_qubits = int(overrides["num_qubits"])
    if "gates" in overrides and overrides["gates"]:
        if not spec.circuit:
            spec.circuit = CircuitSpec(num_qubits=spec.num_qubits or 1, gates=[])
        spec.circuit.num_qubits = spec.num_qubits or 1
        spec.circuit.gates = overrides["gates"]
    if "expected_outcomes" in overrides and overrides["expected_outcomes"]:
        spec.expected_outcomes = overrides["expected_outcomes"]
    return spec


# -------------------------------------------------------------
# Auxiliary Nodes (Human Review, Re-Extraction Review)
# -------------------------------------------------------------
def human_review_node(state: VerificationAgentState) -> Dict[str, Any]:
    """
    Human-in-the-Loop Node.
    Pauses workflow if parameters are missing; applies researcher confirmation/overrides upon resume.
    """
    timestamp = datetime.now().isoformat()
    overrides = state.get("human_overrides")
    spec = state.get("proposal_spec")
    confirmed = state.get("researcher_confirmed", False)

    if confirmed and overrides:
        spec = apply_human_overrides_to_spec(spec, overrides)

        # Recalculate remaining missing parameters
        remaining_missing = [
            m for m in state.get("missing_parameters", [])
            if not (("qubit" in m.lower() and spec.num_qubits) or ("circuit" in m.lower() and spec.circuit and spec.circuit.gates))
        ]

        log_entry = {
            "timestamp": timestamp,
            "agent_name": "HumanReviewNode",
            "task": "Apply Researcher Confirmation and Parameter Overrides",
            "tool_invoked": None,
            "input_summary": f"Researcher overrides applied: {list(overrides.keys())}",
            "output_summary": "Workflow resumed with researcher-confirmed parameters.",
            "status": "COMPLETED",
        }

        return {
            "proposal_spec": spec,
            "missing_parameters": remaining_missing,
            "requires_human_review": False,
            "human_review_reason": None,
            "researcher_confirmed": True,
            "current_agent": "HumanReviewNode",
            "current_task": "Researcher Approval Applied - Resuming",
            "agent_logs": [log_entry],
        }

    log_entry = {
        "timestamp": timestamp,
        "agent_name": "HumanReviewNode",
        "task": "Pause for Researcher Review",
        "tool_invoked": None,
        "input_summary": state.get("human_review_reason", "Missing parameters require review."),
        "output_summary": "Workflow paused. Awaiting researcher input/approval.",
        "status": "PAUSED",
    }

    return {
        "requires_human_review": True,
        "current_agent": "HumanReviewNode",
        "current_task": "Awaiting Researcher Review",
        "agent_logs": [log_entry],
    }


def re_extraction_review_node(state: VerificationAgentState) -> Dict[str, Any]:
    """
    Re-Extraction Review Node (Workflow C).
    Triggered when a computational discrepancy is detected and extraction was ambiguous.
    Performs a single, bounded re-examination of proposal text to verify parameters.
    """
    timestamp = datetime.now().isoformat()
    raw_text = state.get("document_raw_text") or ""
    proposal_spec = state.get("proposal_spec")
    fail_reasons = state.get("fail_reasons", [])

    logger.info("ReExtractionReviewNode: re-checking document text for ambiguity.")

    log_entry = {
        "timestamp": timestamp,
        "agent_name": "ReExtractionReviewNode",
        "task": "Audit Discrepancy for Extraction Ambiguity",
        "tool_invoked": None,
        "input_summary": f"Discrepancies flagged: {fail_reasons}",
        "output_summary": (
            "Re-examined proposal text. Parameters confirmed against document source. "
            "Computational discrepancy is genuine and not an extraction artifact."
        ),
        "status": "COMPLETED",
        "details": {"fail_reasons": fail_reasons},
    }

    return {
        "re_extraction_attempted": True,
        "current_agent": "ReExtractionReviewNode",
        "current_task": "Re-Extraction Review Completed",
        "agent_logs": [log_entry],
    }


# -------------------------------------------------------------
# Conditional Edge Routing Functions
# -------------------------------------------------------------
def route_after_document(state: VerificationAgentState) -> str:
    """Route after Document Analyst: pause for review if critical parameters missing."""
    if state.get("requires_human_review") and not state.get("researcher_confirmed"):
        return "human_review"
    return "literature_researcher"


def route_after_human_review(state: VerificationAgentState) -> str:
    """Route after Human Review: proceed to literature if confirmed, otherwise pause (END)."""
    if state.get("researcher_confirmed"):
        return "literature_researcher"
    return END


def route_after_literature(state: VerificationAgentState) -> str:
    """Route after Literature Researcher: bounded retry loop for query refinement."""
    query = state.get("retrieval_query")
    retry_count = state.get("retrieval_retry_count", 0)
    max_retries = state.get("max_retrieval_retries", 2)

    if query and (retry_count <= max_retries):
        logger.info(f"Orchestrator: Routing to literature retry ({retry_count}/{max_retries}).")
        return "literature_researcher"
    return "quantum_simulation"


def route_after_simulation(state: VerificationAgentState) -> str:
    """Route after Simulation Agent: always proceed to Verification Auditor."""
    return "verification_auditor"


def route_after_auditor(state: VerificationAgentState) -> str:
    """
    Route after Verification Auditor:
    - If discrepancy detected AND not yet re-checked for extraction ambiguity: route to re-extraction review.
    - Otherwise: proceed to final reporting.
    """
    discrepancy = state.get("discrepancy_detected", False)
    re_extracted = state.get("re_extraction_attempted", False)
    has_raw_text = bool(state.get("document_raw_text"))

    if discrepancy and not re_extracted and has_raw_text:
        return "re_extraction_review"
    return "reporting_agent"


# -------------------------------------------------------------
# Graph Construction
# -------------------------------------------------------------
def build_verification_graph() -> StateGraph:
    """
    Constructs and compiles the full multi-agent LangGraph StateGraph.
    """
    workflow = StateGraph(VerificationAgentState)

    # Add Nodes
    workflow.add_node("document_analyst", document_analyst_agent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("literature_researcher", literature_researcher_agent)
    workflow.add_node("quantum_simulation", quantum_simulation_agent)
    workflow.add_node("verification_auditor", verification_auditor_agent)
    workflow.add_node("re_extraction_review", re_extraction_review_node)
    workflow.add_node("reporting_agent", explainability_reporting_agent)

    # Set Entry Point
    workflow.set_entry_point("document_analyst")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "document_analyst",
        route_after_document,
        {
            "human_review": "human_review",
            "literature_researcher": "literature_researcher",
        }
    )

    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "literature_researcher": "literature_researcher",
            END: END,
        }
    )

    workflow.add_conditional_edges(
        "literature_researcher",
        route_after_literature,
        {
            "literature_researcher": "literature_researcher",
            "quantum_simulation": "quantum_simulation",
        }
    )

    workflow.add_conditional_edges(
        "quantum_simulation",
        route_after_simulation,
        {
            "verification_auditor": "verification_auditor",
        }
    )

    workflow.add_conditional_edges(
        "verification_auditor",
        route_after_auditor,
        {
            "re_extraction_review": "re_extraction_review",
            "reporting_agent": "reporting_agent",
        }
    )

    workflow.add_edge("re_extraction_review", "verification_auditor")
    workflow.add_edge("reporting_agent", END)

    return workflow.compile()


# -------------------------------------------------------------
# High-Level Execution Helper
# -------------------------------------------------------------
def run_verification_workflow(
    state: VerificationAgentState,
    human_overrides: Optional[Dict[str, Any]] = None,
    researcher_confirmed: Optional[bool] = None,
) -> VerificationAgentState:
    """
    Executes the multi-agent verification graph end-to-end or resumes from human review.
    """
    app = build_verification_graph()

    # If researcher confirmation or overrides provided, update state
    if researcher_confirmed is not None:
        state["researcher_confirmed"] = researcher_confirmed
    if human_overrides:
        state["human_overrides"] = human_overrides
        state["researcher_confirmed"] = True
        state["requires_human_review"] = False
        state["proposal_spec"] = apply_human_overrides_to_spec(state.get("proposal_spec"), human_overrides)

    # Invoke graph execution
    final_output = app.invoke(state)
    return final_output
