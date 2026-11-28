"""
Candidate research gaps and their novelty investigations (Module 4C / 5).

A gap is never "validated" by the system. It carries the verified statements it came from, the research
dimensions that define it (each tagged with where it came from), and, once checked, a recorded novelty
investigation whose conclusion is rule-based and worded "in the searched literature".
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DIMENSIONS = ("phenomenon", "method", "platform", "noise", "metric", "qubit_scale")
CORE_DIMENSIONS = ("phenomenon", "method", "platform", "noise", "qubit_scale")   # metric alone never defines a question
GENERIC_VALUES = {"entanglement", "fidelity"}      # dropped from a dimension that also names something specific

ORIGIN_LABELS: Dict[str, str] = {
    "stated_limitation": "Methodological limitation",
    "unresolved_problem": "Unresolved problem",
    "future_work": "Unresolved future work",
    "unexplored_combination": "Untried combination in this corpus",
    "benchmark": "Benchmark: a published paper's own question",
}

CLASSIFICATION_LABELS: Dict[str, str] = {
    "already_investigated": "Already investigated",
    "partially_investigated": "Partially investigated",
    "conflicting_findings": "Conflicting research findings",
    "potentially_unexplored": "Potentially unexplored in the searched literature",
    "insufficient_coverage": "Insufficient search coverage",
}

DimSource = Literal["statement", "paper_context", "corpus", "llm"]


class GapDimension(BaseModel):
    values: List[str] = Field(default_factory=list, description="Canonical ontology concepts, or qubit counts as strings.")
    source: DimSource = Field(..., description="statement: in the gap's own evidence; paper_context: from the source paper; "
                                               "corpus: from cross-paper coverage; llm: model interpretation.")


class GapEvidenceItem(BaseModel):
    publication_id: Optional[int] = None
    title: str
    quote: str
    kind: Literal["direct", "interpretation", "inference"]
    page: Optional[int] = None
    role: Literal["supporting", "related"] = "supporting"


class Gap(BaseModel):
    id: Optional[int] = None
    key: str = Field(..., description="Stable de-duplication key; re-running generation updates rather than duplicates.")
    title: str
    statement: str
    research_question: Optional[str] = None
    research_question_source: Optional[str] = Field(None, description="'template' or provider/model when rewritten.")
    origin: str
    dimensions: Dict[str, GapDimension] = Field(default_factory=dict)
    evidence: List[GapEvidenceItem] = Field(default_factory=list)
    generator: str = "deterministic"
    classification: Optional[str] = None
    review_status: str = "candidate"

    def core(self) -> Dict[str, List[str]]:
        """
        The dimensions that define the question. The gap's own dimensions come first; dimensions borrowed from the
        source paper only top up to two. Otherwise a long source paper's context over-specifies the gap, no prior
        work can match it, and the check drifts towards false novelty.
        """
        present = {}
        for d in CORE_DIMENSIONS:
            vals = self.dimensions[d].values if d in self.dimensions else []
            specific = [v for v in vals if v not in GENERIC_VALUES]
            if vals:
                present[d] = specific or vals
        out = {d: v for d, v in present.items() if self.dimensions[d].source != "paper_context"}
        for d, v in present.items():
            if len(out) >= 2:
                break
            out.setdefault(d, v)
        return {d: out[d] for d in CORE_DIMENSIONS if d in out}

    @property
    def supporting_ids(self) -> List[int]:
        return sorted({e.publication_id for e in self.evidence if e.publication_id and e.role == "supporting"})
