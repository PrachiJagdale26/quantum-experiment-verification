"""
Pydantic schemas for research-gap identification and experiment planning.

Pipeline: ResearchPaperAnalysis (per paper) -> ResearchGap (across papers)
          -> ExperimentPlan (per gap) -> VerificationReport (existing engine).
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .proposal import ProposalSpecification

GapType = Literal[
    "stated_limitation",
    "future_work",
    "unexplored_combination",
    "scale_limit",
    "conflicting_results",
]

GAP_TYPE_LABELS: Dict[str, str] = {
    "stated_limitation": "Stated limitation",
    "future_work": "Future work",
    "unexplored_combination": "Unexplored combination",
    "scale_limit": "Scale limit",
    "conflicting_results": "Conflicting results",
}


class GapEvidence(BaseModel):
    """A traceable piece of literature that supports the existence of a gap."""
    paper_title: str
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    quote: str = Field(..., description="Sentence or fact from the paper that reveals the gap.")


class ResearchGap(BaseModel):
    """An open research opportunity identified across the literature."""
    gap_id: str
    title: str = Field(..., description="Short name for the gap.")
    statement: str = Field(..., description="One or two sentences describing what is missing.")
    gap_type: GapType
    evidence: List[GapEvidence] = Field(default_factory=list)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0, description="1 - similarity to existing indexed literature.")
    feasibility: float = Field(default=0.5, ge=0.0, le=1.0, description="How readily it can be tested on a small simulator.")
    evidence_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="How well the literature supports the gap.")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Weighted overall ranking score.")
    testable_on_simulator: bool = False
    # Structured hints for the planner (e.g. system, algorithm, qubits, circuit family).
    hints: Dict[str, str] = Field(default_factory=dict)


class ExperimentPlan(BaseModel):
    """A concrete experiment designed to address a research gap."""
    plan_id: str
    gap_id: str
    title: str
    hypothesis: str
    rationale: str
    method_steps: List[str] = Field(default_factory=list)
    variables: Dict[str, str] = Field(default_factory=dict, description="Independent / dependent / controlled variables.")
    baseline: str = ""
    success_metrics: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    resources: str = ""
    proposal: ProposalSpecification = Field(..., description="Executable specification checked by the verification engine.")
    verification_status: Optional[str] = Field(default=None, description="pass / fail / hold once verified.")
    revision: int = 0
    revision_note: Optional[str] = None
    generated_by: Literal["llm", "template"] = "template"
