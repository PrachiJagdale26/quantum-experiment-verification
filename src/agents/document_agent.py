"""
Document Analyst Agent for LangGraph Quantum Verification.
Analyzes uploaded documents (PDF/DOCX) using Gemini/Vertex AI and strict deterministic parsers.
Extracts structured experiment specifications, checks for missing parameters, and pauses for
human review when critical parameters are absent, without inventing or hallucinating data.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from .state import VerificationAgentState
from .tools import extract_document_tool
from ..schemas.proposal import ProposalSpecification

logger = logging.getLogger(__name__)


def document_analyst_agent(state: VerificationAgentState) -> Dict[str, Any]:
    """
    Agent 1: Document Analyst
    Inspects document text/bytes, executes extraction, flags missing parameters,
    and sets human-review triggers if essential parameters are absent.
    """
    timestamp = datetime.now().isoformat()
    raw_text = state.get("document_raw_text") or ""
    filename = state.get("document_filename") or "proposal.pdf"
    existing_spec = state.get("proposal_spec")
    file_bytes = state.get("extracted_doc_metadata", {}).get("file_bytes")

    logger.info(f"DocumentAnalystAgent executing on '{filename}'")

    # If proposal_spec is already confirmed and provided, inspect it directly
    if existing_spec is not None and state.get("researcher_confirmed", False):
        proposal_spec = existing_spec
        doc_type = state.get("doc_type") or "experiment_proposal"
        paper_analysis = state.get("paper_analysis")
        tool_used = "direct_spec_pass_through"
    else:
        # Run extraction tool
        tool_res = extract_document_tool(
            raw_text=raw_text,
            filename=filename,
            file_bytes=file_bytes,
        )
        proposal_spec = tool_res.get("proposal_spec")
        paper_analysis = tool_res.get("paper_analysis")
        doc_type = tool_res.get("doc_type", "experiment_proposal")
        tool_used = "extract_document_tool"

    # Analyze missing parameters and ambiguities
    missing_parameters: List[str] = []
    inconsistencies: List[str] = []

    if proposal_spec is None:
        missing_parameters.append("Experiment specification could not be extracted.")
    else:
        if proposal_spec.num_qubits is None or proposal_spec.num_qubits <= 0:
            missing_parameters.append("Number of qubits is unspecified or invalid.")
        
        if not proposal_spec.circuit or not proposal_spec.circuit.gates:
            missing_parameters.append("Quantum circuit gate sequence is missing.")
        else:
            # Check for gates targeting out-of-range qubits
            if proposal_spec.num_qubits:
                for g in proposal_spec.circuit.gates:
                    targets = getattr(g, "target_qubits", []) or []
                    for target in targets:
                        if target >= proposal_spec.num_qubits or target < 0:
                            inconsistencies.append(
                                f"Gate '{g.name}' targets qubit {target}, which exceeds register size {proposal_spec.num_qubits}."
                            )

        if not proposal_spec.expected_outcomes:
            missing_parameters.append("Expected measurement probabilities are unspecified.")
        else:
            prob_sum = sum(o.expected_probability for o in proposal_spec.expected_outcomes)
            if abs(prob_sum - 1.0) > 0.05:
                inconsistencies.append(f"Expected outcome probabilities sum to {prob_sum:.4f} (expected ~1.0).")

        if proposal_spec.missing_information:
            for item in proposal_spec.missing_information:
                if item not in missing_parameters:
                    missing_parameters.append(item)

    # Determine whether human review is required before proceeding to simulation
    # Critical parameters: qubit count and circuit gates
    critical_missing = any(
        "qubit" in p.lower() or "circuit" in p.lower() or "gate" in p.lower()
        for p in missing_parameters
    )
    
    requires_human_review = critical_missing and not state.get("researcher_confirmed", False)
    human_review_reason = None
    if requires_human_review:
        human_review_reason = (
            f"Extraction identified critical missing parameters: {', '.join(missing_parameters)}. "
            "Researcher approval or parameter input is required before simulation."
        )

    log_entry = {
        "timestamp": timestamp,
        "agent_name": "DocumentAnalystAgent",
        "task": "Extract and Validate Experiment Specification",
        "tool_invoked": tool_used,
        "input_summary": f"Document '{filename}' ({len(raw_text)} chars)",
        "output_summary": (
            f"Extracted: '{proposal_spec.title if proposal_spec else 'None'}', "
            f"Qubits: {proposal_spec.num_qubits if proposal_spec else 'None'}, "
            f"Missing params: {len(missing_parameters)}, "
            f"Requires review: {requires_human_review}"
        ),
        "status": "PAUSED" if requires_human_review else "COMPLETED",
        "details": {
            "missing_parameters": missing_parameters,
            "inconsistencies": inconsistencies,
            "doc_type": doc_type,
        },
    }

    return {
        "proposal_spec": proposal_spec,
        "paper_analysis": paper_analysis,
        "doc_type": doc_type,
        "missing_parameters": missing_parameters,
        "potential_inconsistencies": inconsistencies,
        "requires_human_review": requires_human_review,
        "human_review_reason": human_review_reason,
        "current_agent": "DocumentAnalystAgent",
        "current_task": "Document Analysis Completed" if not requires_human_review else "Awaiting Researcher Input",
        "agent_logs": [log_entry],
    }
