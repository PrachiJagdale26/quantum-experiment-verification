"""
Verification Auditor Agent for LangGraph Quantum Verification.
Applies deterministic verification rules across proposal specs, Qiskit simulation outputs,
and literature evidence. Enforces the strict scientific rule: computational FAIL results
cannot be overturned or overridden by LLM hypotheses or literature arguments.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging

from .state import VerificationAgentState
from .tools import run_verification_audit_tool
from ..schemas.verification import CheckStatus

logger = logging.getLogger(__name__)


def verification_auditor_agent(state: VerificationAgentState) -> Dict[str, Any]:
    """
    Agent 4: Verification Auditor
    Deterministically audits the proposal using quantum verification rules.
    Detects computational discrepancies and ensures simulation failures remain strictly FAIL.
    """
    timestamp = datetime.now().isoformat()
    proposal_spec = state.get("proposal_spec")
    sim_output = state.get("simulation_output")
    outcome_comparisons = state.get("outcome_comparisons", [])
    citations = state.get("literature_citations", [])
    tvd_score = state.get("tvd_score")

    if not proposal_spec:
        logger.error("VerificationAuditor: No proposal specification to audit.")
        return {
            "verification_checks": [],
            "summary_counts": {"total": 0, "passed": 0, "failed": 1, "warning": 0, "unsupported": 0},
            "overall_status": "FAIL",
            "discrepancy_detected": True,
            "fail_reasons": ["No experiment proposal specification found in state."],
            "current_agent": "VerificationAuditorAgent",
            "current_task": "Verification Failed - Missing Spec",
            "agent_logs": [{
                "timestamp": timestamp,
                "agent_name": "VerificationAuditorAgent",
                "task": "Audit Verification Rules",
                "tool_invoked": None,
                "input_summary": "Missing proposal specification",
                "output_summary": "Verification audit failed due to missing specification.",
                "status": "FAILED",
            }],
        }

    # Execute deterministic verification audit tool
    tool_res = run_verification_audit_tool(
        proposal_spec=proposal_spec,
        simulation_output=sim_output,
        outcome_comparisons=outcome_comparisons,
        citations=citations,
    )

    checks = tool_res.get("checks", [])
    summary_counts = tool_res.get("summary_counts", {})
    overall_status = tool_res.get("overall_status", "FAIL")
    fail_reasons = tool_res.get("fail_reasons", [])

    # Discrepancy detection: any simulation or distribution mismatch
    has_sim_mismatch = any(
        c.name == "Measurement Outcome Distribution" and c.status == CheckStatus.FAIL
        for c in checks
    )
    has_syntax_fail = any(
        c.name in ("Qubit Count Validation", "Circuit Syntax & Valid Gate Operations") and c.status == CheckStatus.FAIL
        for c in checks
    )
    
    discrepancy_detected = has_sim_mismatch or has_syntax_fail or (summary_counts.get("failed", 0) > 0)

    # CRITICAL SCIENTIFIC RULE ENFORCEMENT:
    # A computational simulation failure is inviolable.
    if has_sim_mismatch or summary_counts.get("failed", 0) > 0:
        overall_status = "FAIL"
        logger.warning(
            f"VerificationAuditor: Computational discrepancy confirmed ({fail_reasons}). "
            "Inviolable rule applied: LLM cannot override computational FAIL."
        )

    log_entry = {
        "timestamp": timestamp,
        "agent_name": "VerificationAuditorAgent",
        "task": "Execute Deterministic Verification Rules",
        "tool_invoked": "run_verification_audit_tool",
        "input_summary": (
            f"Auditing '{proposal_spec.title}', "
            f"{len(citations)} citations, "
            f"Simulated: {sim_output is not None}"
        ),
        "output_summary": (
            f"Status: {overall_status} "
            f"({summary_counts.get('passed', 0)} passed, {summary_counts.get('failed', 0)} failed, "
            f"{summary_counts.get('warning', 0)} warnings). "
            f"Discrepancy detected: {discrepancy_detected}"
        ),
        "status": "COMPLETED",
        "details": {
            "overall_status": overall_status,
            "summary_counts": summary_counts,
            "fail_reasons": fail_reasons,
            "discrepancy_detected": discrepancy_detected,
        },
    }

    return {
        "verification_checks": checks,
        "summary_counts": summary_counts,
        "overall_status": overall_status,
        "discrepancy_detected": discrepancy_detected,
        "fail_reasons": fail_reasons,
        "current_agent": "VerificationAuditorAgent",
        "current_task": f"Audit Completed ({overall_status})",
        "agent_logs": [log_entry],
    }
