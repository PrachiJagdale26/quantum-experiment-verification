"""
Research proposal for a selected gap (Module 6).

Every parameter states where it comes from (the literature with a verified quote, the gap's own evidence,
an explicit assumption, or the researcher). Literature justifications cite publications by id and their quotes
are checked against the stored text. Anything the system cannot ground is listed in missing_information
instead of being invented. The system never approves a proposal; only the researcher does.
"""

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from ..schemas.proposal import ExpectedOutcome, GateOperation

FeasibilityClass = Literal["simulatable", "needs_hardware", "needs_physical_validation"]
FEASIBILITY_LABELS = {
    "simulatable": "Simulatable",
    "needs_hardware": "Requires quantum hardware",
    "needs_physical_validation": "Requires additional physical validation",
}
FEASIBILITY_ORDER = ["simulatable", "needs_hardware", "needs_physical_validation"]   # increasing demand

ParamSource = Literal["literature", "gap_evidence", "assumption", "researcher"]
PARAM_SOURCE_LABELS = {"literature": "From the literature (quote verified)", "gap_evidence": "From the gap's evidence",
                       "assumption": "Assumption", "researcher": "Set by the researcher"}


class Parameter(BaseModel):
    name: str
    value: str
    unit: str = ""
    source: ParamSource
    basis: str = Field("", description="The verified quote, or why the value was assumed.")
    publication_id: Optional[int] = None
    verified: bool = False


class Justification(BaseModel):
    publication_id: int
    title: str
    year: Optional[int] = None
    doi: Optional[str] = None
    role: Literal["motivates", "closest_prior_work", "baseline", "method_source"]
    point: str = Field(..., description="What this paper contributes to the argument.")
    quote: Optional[str] = None
    verified: bool = False


class CircuitDraft(BaseModel):
    num_qubits: int = Field(..., ge=1, le=10)
    gates: List[GateOperation]
    expected_outcomes: List[ExpectedOutcome] = Field(default_factory=list)
    prediction_basis: str = Field("", description="Why these outcomes are expected (theory), never the simulator's output.")
    expected_state: Optional[Dict[str, List[float]]] = Field(
        None, description="Optional target pure state {bitstring: [re, im]} for the state-fidelity check.")
    expected_outcomes_noisy: List[ExpectedOutcome] = Field(
        default_factory=list, description="Optional predictions under the plan's stated noise (checked against noisy simulation).")

    @model_validator(mode="after")
    def _qubits_in_register(self) -> "CircuitDraft":
        for i, g in enumerate(self.gates, 1):
            bad = [q for q in g.target_qubits if q >= self.num_qubits]
            if bad:
                raise ValueError(f"gate {i} ({g.name}) uses qubit {bad[0]}, outside the {self.num_qubits}-qubit register")
        state_keys = [ExpectedOutcome(basis_state=b, expected_probability=0) for b in (self.expected_state or {})]
        for o in list(self.expected_outcomes) + list(self.expected_outcomes_noisy) + state_keys:
            if len(o.basis_state) != self.num_qubits or set(o.basis_state) - {"0", "1"}:
                raise ValueError(f"outcome '{o.basis_state}' is not a {self.num_qubits}-bit string")
        return self


class ResearchProposal(BaseModel):
    gap_id: int
    gap_statement: str
    title: str
    research_problem: str = ""
    research_question: str = ""
    objectives: List[str] = Field(default_factory=list)
    hypothesis: str = ""
    motivation: str = ""
    literature_justification: List[Justification] = Field(default_factory=list)
    methodology: List[str] = Field(default_factory=list)
    quantum_system: str = ""
    techniques: List[str] = Field(default_factory=list)
    independent_variables: List[str] = Field(default_factory=list)
    dependent_variables: List[str] = Field(default_factory=list)
    controlled_variables: List[str] = Field(default_factory=list)
    control_conditions: List[str] = Field(default_factory=list)
    parameters: List[Parameter] = Field(default_factory=list)
    expected_observations: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    baselines: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    expected_contribution: str = ""
    why_addresses_gap: str = ""
    feasibility_class: FeasibilityClass = "simulatable"
    feasibility_reason: str = ""
    circuit: Optional[CircuitDraft] = None
    missing_information: List[str] = Field(default_factory=list)
    generated_by: str = "template"
    warnings: List[str] = Field(default_factory=list)


# (field, label, kind) in the order Module 6 lists them; kind: text | list | special
SECTIONS: List[Tuple[str, str, str]] = [
    ("gap_statement", "Research gap", "text"), ("research_problem", "Research problem", "text"),
    ("research_question", "Research question", "text"), ("objectives", "Research objectives", "list"),
    ("hypothesis", "Hypothesis", "text"), ("motivation", "Scientific motivation", "text"),
    ("literature_justification", "Literature justification", "special"), ("methodology", "Proposed methodology", "list"),
    ("quantum_system", "Quantum system", "text"), ("techniques", "Quantum algorithms or techniques", "list"),
    ("independent_variables", "Independent variables", "list"), ("dependent_variables", "Dependent variables", "list"),
    ("controlled_variables", "Controlled variables", "list"), ("control_conditions", "Control conditions", "list"),
    ("parameters", "Experimental parameters", "special"), ("expected_observations", "Expected observations", "list"),
    ("metrics", "Evaluation metrics", "list"), ("baselines", "Baseline methods", "list"),
    ("challenges", "Potential challenges", "list"), ("limitations", "Limitations", "list"),
    ("expected_contribution", "Expected contribution", "text"), ("why_addresses_gap", "Why this addresses the gap", "text"),
]
EDITABLE_TEXT = [f for f, _, k in SECTIONS if k == "text" and f != "gap_statement"] + ["title"]
EDITABLE_LIST = [f for f, _, k in SECTIONS if k == "list"]
REQUIRED = ["research_question", "hypothesis", "methodology", "dependent_variables", "metrics"]
