"""
Shared typed state definitions for the LangGraph Multi-Agent Quantum Verification System.
Maintains full end-to-end traceability across document analysis, literature retrieval,
quantum circuit simulation, verification auditing, and report generation.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
import operator
from datetime import datetime

from ..schemas.proposal import ProposalSpecification, ExtractedClaim
from ..schemas.research_paper import ResearchPaperAnalysis
from ..schemas.verification import (
    LiteratureCitation,
    SimulationOutput,
    OutcomeComparison,
    VerificationCheck,
    VerificationReport,
    CheckStatus,
)


def append_logs(
    existing: Optional[List[Dict[str, Any]]],
    new: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Reducer to safely append execution logs across agent steps."""
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new


class AgentLogEntry(TypedDict, total=False):
    timestamp: str
    agent_name: str
    task: str
    tool_invoked: Optional[str]
    input_summary: Optional[str]
    output_summary: Optional[str]
    status: Literal["RUNNING", "COMPLETED", "FAILED", "PAUSED"]
    details: Optional[Dict[str, Any]]


class VerificationAgentState(TypedDict, total=False):
    """
    Shared typed state maintained across all agents in the verification graph.
    """
    # 1. Uploaded Document Information
    document_raw_text: Optional[str]
    document_filename: Optional[str]
    document_file_type: Optional[str]
    doc_type: Literal["experiment_proposal", "research_paper", "unknown"]
    extracted_doc_metadata: Optional[Dict[str, Any]]

    # 2. Extracted Scientific Specifications
    proposal_spec: Optional[ProposalSpecification]
    paper_analysis: Optional[ResearchPaperAnalysis]
    extracted_claims: List[ExtractedClaim]

    # 3. Missing Parameters & Human-in-the-Loop Controls
    missing_parameters: List[str]
    potential_inconsistencies: List[str]
    requires_human_review: bool
    human_review_reason: Optional[str]
    researcher_confirmed: bool
    human_overrides: Optional[Dict[str, Any]]

    # 4. Literature Evidence & Provenance
    literature_citations: List[LiteratureCitation]
    literature_evaluations: List[Dict[str, Any]]
    retrieval_query: Optional[str]
    retrieval_retry_count: int
    max_retrieval_retries: int
    literature_grounding_passed: bool

    # 5. Independent Quantum Simulation Results (Strictly Computational)
    simulation_output: Optional[SimulationOutput]
    outcome_comparisons: List[OutcomeComparison]
    expected_probabilities: Dict[str, float]
    computed_probabilities: Dict[str, float]
    tvd_score: Optional[float]
    fidelity_score: Optional[float]
    simulation_error: Optional[str]

    # 6. Verification Audit Results
    verification_checks: List[VerificationCheck]
    summary_counts: Dict[str, int]
    overall_status: Optional[str]  # e.g., "PASS", "FAIL", "REQUIRES_REVIEW"
    discrepancy_detected: bool
    re_extraction_attempted: bool
    re_extraction_outcome: Optional[str]  # CONFIRMED | POSSIBLE_EXTRACTION_ARTIFACT | UNDETERMINED
    fail_reasons: List[str]

    # 7. Agent Decisions, Logs & Traceability
    agent_logs: Annotated[List[Dict[str, Any]], append_logs]
    current_agent: Optional[str]
    current_task: Optional[str]
    agent_decisions: Dict[str, Any]

    # 8. Final Report Artifacts
    final_report: Optional[VerificationReport]
    markdown_report: Optional[str]
    errors: List[str]


def create_initial_state(
    raw_text: Optional[str] = None,
    filename: Optional[str] = None,
    doc_type: Literal["experiment_proposal", "research_paper", "unknown"] = "experiment_proposal",
    proposal_spec: Optional[ProposalSpecification] = None,
    researcher_confirmed: bool = False,
) -> VerificationAgentState:
    """Helper to initialize a clean verification agent state."""
    return VerificationAgentState(
        document_raw_text=raw_text,
        document_filename=filename,
        document_file_type=filename.split(".")[-1].lower() if filename else None,
        doc_type=doc_type,
        extracted_doc_metadata={},
        proposal_spec=proposal_spec,
        paper_analysis=None,
        extracted_claims=[],
        missing_parameters=[],
        potential_inconsistencies=[],
        requires_human_review=False,
        human_review_reason=None,
        researcher_confirmed=researcher_confirmed,
        human_overrides=None,
        literature_citations=[],
        literature_evaluations=[],
        retrieval_query=None,
        retrieval_retry_count=0,
        max_retrieval_retries=2,
        literature_grounding_passed=False,
        simulation_output=None,
        outcome_comparisons=[],
        expected_probabilities={},
        computed_probabilities={},
        tvd_score=None,
        fidelity_score=None,
        simulation_error=None,
        verification_checks=[],
        summary_counts={"total": 0, "passed": 0, "failed": 0, "warning": 0, "unsupported": 0},
        overall_status=None,
        discrepancy_detected=False,
        re_extraction_attempted=False,
        re_extraction_outcome=None,
        fail_reasons=[],
        agent_logs=[{
            "timestamp": datetime.now().isoformat(),
            "agent_name": "SystemOrchestrator",
            "task": "Initialize Verification State",
            "tool_invoked": None,
            "input_summary": f"Document: {filename or 'Manual Specification'}",
            "output_summary": "State initialized successfully.",
            "status": "COMPLETED",
        }],
        current_agent="SystemOrchestrator",
        current_task="Ready",
        agent_decisions={},
        final_report=None,
        markdown_report=None,
        errors=[],
    )
