"""
Explainability and Reporting Agent for LangGraph Quantum Verification.
Collects and synthesizes findings across all verification agents into an evidence-backed,
transparent scientific explanation, generating structured JSON and Markdown reports.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from .state import VerificationAgentState
from .tools import generate_report_tool
from ..schemas.verification import CheckStatus

logger = logging.getLogger(__name__)


def explainability_reporting_agent(state: VerificationAgentState) -> Dict[str, Any]:
    """
    Agent 5: Explainability and Reporting Agent
    Collects outputs from all agents, synthesizes grounded scientific explanations,
    and produces comprehensive JSON and Markdown artifacts.
    """
    timestamp = datetime.now().isoformat()
    proposal_spec = state.get("proposal_spec")
    sim_output = state.get("simulation_output")
    checks = state.get("verification_checks", [])
    summary_counts = state.get("summary_counts", {})
    outcome_comparisons = state.get("outcome_comparisons", [])
    citations = state.get("literature_citations", [])
    tvd = state.get("tvd_score")
    fidelity = state.get("fidelity_score")
    overall_status = state.get("overall_status", "UNKNOWN")
    fail_reasons = state.get("fail_reasons", [])

    if not proposal_spec:
        logger.error("ReportingAgent: No proposal spec to generate report for.")
        return {
            "final_report": None,
            "markdown_report": "### Error: No proposal specification available for report generation.",
            "current_agent": "ReportingAgent",
            "current_task": "Reporting Aborted - No Spec",
            "agent_logs": [{
                "timestamp": timestamp,
                "agent_name": "ReportingAgent",
                "task": "Generate Final Verification Report",
                "tool_invoked": None,
                "input_summary": "Missing proposal specification",
                "output_summary": "Aborted report generation.",
                "status": "FAILED",
            }],
        }

    # Construct rigorous scientific explanation notes
    notes_lines = []
    notes_lines.append(f"### Scientific Verification Summary (Status: {overall_status})")
    
    # 1. Computational Simulation Findings
    if sim_output:
        total_prob = sum(getattr(sim_output, "probabilities", {}).values())
        notes_lines.append(
            f"- **Computational Simulation**: Evaluated {proposal_spec.num_qubits}-qubit circuit ({getattr(sim_output, 'execution_time_ms', 0.0):.1f} ms). "
            f"Pure statevector normalized with total probability {total_prob:.4f}."
        )
        if tvd is not None:
            notes_lines.append(f"- **Total Variation Distance (TVD)**: {tvd:.4f} (Threshold <= 0.05).")
        if fidelity is not None:
            notes_lines.append(f"- **Quantum State Fidelity**: {fidelity:.4f} (Threshold >= 0.95).")
    else:
        notes_lines.append("- **Computational Simulation**: Circuit execution was not possible or omitted.")

    # 2. Literature Grounding Findings
    if citations:
        notes_lines.append(
            f"- **Literature Grounding**: Corroborated by {len(citations)} indexed peer-reviewed citations. "
            f"Top evidence source: '{citations[0].filename}' (p. {citations[0].page_number or 1})."
        )
    else:
        notes_lines.append("- **Literature Grounding**: No matching peer-reviewed literature passages found.")

    # 3. Discrepancies and Failures
    if fail_reasons:
        notes_lines.append("\n**Identified Discrepancies & Violations**:")
        for r in fail_reasons:
            notes_lines.append(f"  - ❌ {r}")
        notes_lines.append("\n*Computational Discrepancy Rule*: Simulation failures cannot be overturned by theoretical or textual claims.")
    elif overall_status == "PASS":
        notes_lines.append("\n✅ All deterministic rules, statevector normalization, and measurement probabilities match author claims within tolerance.")

    explainability_notes = "\n".join(notes_lines)

    # Call report generation tool
    tool_res = generate_report_tool(
        proposal_spec=proposal_spec,
        simulation_output=sim_output,
        checks=checks,
        summary_counts=summary_counts,
        outcome_comparisons=outcome_comparisons,
        citations=citations,
        tvd=tvd,
        fidelity=fidelity,
        explainability_notes=explainability_notes,
    )

    report = tool_res.get("report")
    md_str = tool_res.get("markdown_str", "")

    log_entry = {
        "timestamp": timestamp,
        "agent_name": "ReportingAgent",
        "task": "Compile Final Report & Explainability Artifacts",
        "tool_invoked": "generate_report_tool",
        "input_summary": f"Compiled findings for '{proposal_spec.title}'",
        "output_summary": f"Report compiled: {overall_status}, Markdown length: {len(md_str)} chars.",
        "status": "COMPLETED",
        "details": {
            "overall_status": overall_status,
            "markdown_length": len(md_str),
        },
    }

    return {
        "final_report": report,
        "markdown_report": md_str,
        "current_agent": "ReportingAgent",
        "current_task": "Verification Report Completed",
        "agent_logs": [log_entry],
    }
