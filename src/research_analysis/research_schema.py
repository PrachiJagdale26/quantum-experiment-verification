"""
Structured, provenance-carrying representation of a paper (Module 2 / 4A).

Kinds of statement (never mixed):
  direct          the text IS a passage of the source, verified to occur in it
  interpretation  a model's restatement, backed by an excerpt that was verified in the source
  unverified      a model's statement whose excerpt could not be found in the source (low confidence,
                  never used as evidence for a gap)
  inference       a conclusion drawn across several publications (produced by cross-paper analysis)
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

StatementField = Literal[
    "research_problem", "objective", "research_question", "existing_approach", "methodology",
    "quantum_system", "experimental_parameter", "technique", "noise_model", "assumption",
    "evaluation_metric", "dataset_or_environment", "result", "limitation", "unresolved_problem",
    "future_work", "conclusion",
]
STATEMENT_FIELDS: List[str] = list(StatementField.__args__)  # type: ignore[attr-defined]

FIELD_LABELS: Dict[str, str] = {
    "research_problem": "Research problem", "objective": "Objective", "research_question": "Research question",
    "existing_approach": "Existing approaches", "methodology": "Methodology", "quantum_system": "Quantum system",
    "experimental_parameter": "Experimental parameters", "technique": "Algorithms and techniques",
    "noise_model": "Noise model and physical assumptions", "assumption": "Assumptions",
    "evaluation_metric": "Evaluation metrics", "dataset_or_environment": "Dataset or simulation environment",
    "result": "Results", "limitation": "Limitations", "unresolved_problem": "Unresolved problems",
    "future_work": "Future work", "conclusion": "Conclusions",
}

Kind = Literal["direct", "interpretation", "unverified", "inference"]


class Provenance(BaseModel):
    publication_id: int
    source: Literal["abstract", "full_text"]
    excerpt: str = Field(..., description="Text from the source supporting the statement.")
    verified: bool = Field(..., description="True if the excerpt was found in the source text.")
    match_score: float = Field(..., ge=0.0, le=100.0, description="100 = verbatim; fuzzy otherwise.")
    page: Optional[int] = None
    section: Optional[str] = None


class Statement(BaseModel):
    field: StatementField
    text: str
    kind: Kind
    provenance: Optional[Provenance] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    extractor: str = Field(..., description="'rule-based', 'ontology', or provider/model for LLM output.")


class ResearchDimensions(BaseModel):
    """Deterministic tags from the paper's own wording (ontology), used for cross-paper comparison."""
    platforms: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    noise: List[str] = Field(default_factory=list)
    phenomena: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    qubit_counts: List[int] = Field(default_factory=list)


class PaperAnalysisRecord(BaseModel):
    publication_id: int
    title: str
    analysed_from: Literal["abstract", "full_text"]
    llm: Optional[str] = Field(None, description="provider/model if a live model contributed; None if rule-based only.")
    statements: List[Statement] = Field(default_factory=list)
    dimensions: ResearchDimensions = Field(default_factory=ResearchDimensions)
    warnings: List[str] = Field(default_factory=list)

    def by_field(self) -> Dict[str, List[Statement]]:
        out: Dict[str, List[Statement]] = {}
        for s in self.statements:
            out.setdefault(s.field, []).append(s)
        return out

    @property
    def verified_share(self) -> float:
        llm_items = [s for s in self.statements if s.extractor not in ("rule-based", "ontology")]
        return sum(1 for s in llm_items if s.provenance and s.provenance.verified) / len(llm_items) if llm_items else 1.0
