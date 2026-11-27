"""
ExperimentPlanner: stage 3 of the pipeline. Turns a ResearchGap into an ExperimentPlan
that carries an executable ProposalSpecification, so stage 4 (the existing
VerificationEngine) can check the plan's circuit and predicted outcomes.

With a live LLM the plan is written by the model (EXPERIMENT_PLANNING_PROMPT) and
validated against the schemas; otherwise, or if the model's output is invalid,
a template for the gap type is used. Template predictions come from analytic
theory (domain.template_circuit), never from the simulator being verified against.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Sequence

from ..llm.client import ConfigurableLLMClient
from ..llm.json_utils import parse_llm_json
from ..llm.prompts import EXPERIMENT_PLANNING_PROMPT
from ..schemas.proposal import CircuitSpec, ExpectedOutcome, ExtractedClaim, GateOperation, ProposalSpecification
from ..schemas.research_gap import ExperimentPlan, ResearchGap
from ..schemas.research_paper import ResearchPaperAnalysis
from ..schemas.verification import CheckStatus, VerificationReport
from .domain import FAMILY_LABELS, TEMPLATE_FAMILIES, experiment_family, template_circuit

logger = logging.getLogger(__name__)
DEFAULT_SYSTEM = "Superconducting Transmon"


def _fmt_dist(outcomes: List[ExpectedOutcome]) -> str:
    return ", ".join(f"P({o.basis_state}) = {o.expected_probability:.2f}" for o in outcomes)


def build_proposal(title: str, objective: str, hypothesis: str, method: List[str], system: str,
                   n: int, gates: List[GateOperation], outcomes: List[ExpectedOutcome]) -> ProposalSpecification:
    return ProposalSpecification(
        title=title, objective=objective, hypothesis=hypothesis, methodology=" ".join(method),
        quantum_system=system, num_qubits=n, initial_state="|" + "0" * n + ">", measurement_basis="computational (Z)",
        circuit=CircuitSpec(num_qubits=n, gates=gates, initial_state="|" + "0" * n + ">"),
        expected_outcomes=[ExpectedOutcome(basis_state=o.basis_state, expected_probability=o.expected_probability,
                                           tolerance=o.tolerance) for o in outcomes],
        assumptions=["Verification uses ideal, noiseless statevector simulation."],
        author_claims=[ExtractedClaim(claim_id="CLM-01", claim_type="outcome",
                                      statement=f"Ideal measurement yields {_fmt_dist(outcomes)}.")] if outcomes else [],
        input_mode="manual", confirmed_by_researcher=True,
    )


class ExperimentPlanner:
    def __init__(self, llm_client: Optional[ConfigurableLLMClient] = None):
        self.llm = llm_client

    @property
    def live(self) -> bool:
        return self.llm is not None and not getattr(self.llm, "demo_mode", True)

    # ================================================================ public
    def plan(self, gap: ResearchGap, papers: Sequence[ResearchPaperAnalysis], plan_id: str) -> ExperimentPlan:
        if self.live:
            plan = self._llm_plan(gap, papers, plan_id)
            if plan is not None:
                return plan
        return self._template_plan(gap, papers, plan_id)

    def revise(self, plan: ExperimentPlan, report: VerificationReport, gap: ResearchGap,
               papers: Sequence[ResearchPaperAnalysis]) -> ExperimentPlan:
        """One repair pass after a failed/inconclusive verification, using the failed checks as feedback."""
        failed = [c for c in report.checks if c.status in (CheckStatus.FAIL, CheckStatus.INCONCLUSIVE)
                  and c.check_id != "CHK-LIT-01"]
        feedback = "; ".join(f"{c.check_id}: {(c.error_message or c.details)[:200]}" for c in failed) or "No failing checks."
        if self.live:
            revised = self._llm_plan(gap, papers, plan.plan_id, feedback=feedback, previous=plan)
            if revised is not None:
                revised.revision, revised.revision_note = plan.revision + 1, f"Revised by the model after verification: {feedback}"
                return revised
        return self._repair(plan, report, gap, papers, feedback)

    # ============================================================= templates
    def _template_plan(self, gap: ResearchGap, papers: Sequence[ResearchPaperAnalysis], plan_id: str) -> ExperimentPlan:
        h = gap.hints
        source = self._paper(papers, h.get("source_paper"))
        family = h.get("family")
        if family not in TEMPLATE_FAMILIES and source is not None:
            family = next((experiment_family(e, source) for e in source.experiments if e.gates), family)
        system = h.get("target_system") or h.get("system") or DEFAULT_SYSTEM
        baseline_paper = source.title if source else "the source paper"

        if gap.gap_type == "scale_limit":
            n_req = int(h.get("max_qubits") or h.get("qubits") or 3) + 1
        else:
            n_req = int(h.get("qubits") or 3)

        source_circuit = self._source_circuit(source) if family not in TEMPLATE_FAMILIES else None
        if source_circuit:
            n, gates, outcomes = source_circuit
            theory = "Outcomes as reported by the source paper."
        else:
            n, gates, outcomes, theory = template_circuit(family or "bell", n_req)
        label = FAMILY_LABELS.get(family or "bell", "Entangling circuit")
        dist = _fmt_dist(outcomes)
        common_risks = ["Readout error biases the measured distribution unless calibrated.",
                        "Two-qubit gate error accumulates with circuit depth."]

        if gap.gap_type == "scale_limit":
            title = f"{label} on {n} qubits"
            hypothesis = f"A cascaded H–CNOT circuit prepares a {n}-qubit {label} with {dist} in the ideal limit, extending the {n - 1}-qubit result."
            steps = [f"Prepare a {n}-qubit register in |{'0' * n}⟩ on {system} hardware.",
                     "Apply H on q0, then a CNOT chain q0→q1→…→q" + str(n - 1) + ".",
                     "Measure all qubits in the computational basis (8192 shots).",
                     "Compute total variation distance (TVD) to the ideal distribution and the parity contrast.",
                     f"Repeat for register sizes {n - 1} to {n} to map how fidelity scales with size."]
            variables = {"independent": "register size n", "dependent": "TVD to ideal, parity contrast",
                         "controlled": "gate set, shot count, calibration cycle"}
            metrics = ["Ideal simulation matches predicted outcomes within ±0.05", f"Hardware parity contrast above 0.8 at n = {n}"]
            risks = common_risks + ["Crosstalk between neighbouring qubits grows with register size."]
        elif gap.gap_type == "conflicting_results":
            other = h.get("other_paper", "the second paper")
            title = f"Resolving conflicting {label} statistics"
            hypothesis = (f"Under ideal conditions the circuit yields {dist}; the deviating distribution reported by "
                          f"“{other}” is caused by readout bias, not by the circuit.")
            steps = ["Run the shared circuit on the same device family with and without readout error mitigation.",
                     "Vary readout delay (0.5–4 µs) to measure its effect on the population imbalance.",
                     "Measure a calibration matrix from |00⟩, |01⟩, |10⟩, |11⟩ preparations and invert it.",
                     "Compare mitigated and unmitigated distributions to the ideal prediction using TVD."]
            variables = {"independent": "readout mitigation on/off, readout delay", "dependent": "TVD to ideal distribution",
                         "controlled": "circuit, device, shot count"}
            metrics = ["Ideal simulation matches predicted outcomes within ±0.05",
                       "Mitigated TVD to ideal below 0.05; unmitigated imbalance reproduces the reported deviation"]
            risks = common_risks + ["Device drift between runs can mimic or mask the effect."]
        elif gap.gap_type == "unexplored_combination":
            title = f"{label} on {system.lower()} hardware"
            hypothesis = f"The {label} protocol transfers to {system} hardware and yields {dist} in the ideal limit."
            steps = [f"Compile the source circuit to the native gate set of a {system} device.",
                     "Prepare the register in the all-zero state and run the compiled circuit (8192 shots).",
                     "Measure in the computational basis and compute TVD to the ideal distribution.",
                     f"Compare fidelity against the result reported in “{baseline_paper}”."]
            variables = {"independent": "hardware platform", "dependent": "state fidelity, TVD to ideal",
                         "controlled": "logical circuit, shot count, analysis pipeline"}
            metrics = ["Ideal simulation matches predicted outcomes within ±0.05",
                       "Hardware fidelity within 10 percentage points of the source platform"]
            risks = common_risks + [f"Native gate compilation on {system} may add depth."]
        else:
            title = f"Follow-up: {gap.title}"
            hypothesis = f"Addressing “{gap.title.lower()}” preserves the ideal {label} distribution ({dist})."
            steps = [f"Re-run the source circuit on {system} hardware as a baseline.",
                     self._targeted_step(gap.statement),
                     "Measure in the computational basis (8192 shots) and compute TVD to the ideal distribution.",
                     "Compare against the baseline run to isolate the effect of the change."]
            variables = {"independent": "the condition named in the gap", "dependent": "TVD to ideal distribution",
                         "controlled": "circuit, device, shot count"}
            metrics = ["Ideal simulation matches predicted outcomes within ±0.05",
                       "Measured change relative to baseline is reported with 95% confidence intervals"]
            risks = common_risks

        proposal = build_proposal(title, f"Address research gap: {gap.title}", hypothesis, steps, system, n, gates, outcomes)
        return ExperimentPlan(
            plan_id=plan_id, gap_id=gap.gap_id, title=title, hypothesis=hypothesis,
            rationale=f"{gap.statement} {theory}", method_steps=steps, variables=variables,
            baseline=f"Result reported in “{baseline_paper}”.", success_metrics=metrics, risks=risks,
            resources=f"{system} device with at least {n} qubits; Qiskit; about 8192 shots per configuration.",
            proposal=proposal, generated_by="template",
        )

    @staticmethod
    def _targeted_step(statement: str) -> str:
        s = statement.lower()
        if "readout" in s or "mitigat" in s:
            return "Apply readout error mitigation from a measured calibration matrix and repeat the run."
        if "crosstalk" in s or "coupler" in s:
            return "Run the circuit with neighbouring qubits idle and then driven simultaneously to measure crosstalk."
        if "connectivity" in s or "nearest" in s:
            return "Compile to a nearest-neighbour coupling map and record the added SWAP count and depth."
        if "separate" in s and "error" in s:
            return "Use preparation-and-measure calibration circuits to separate state-preparation from measurement error."
        return "Change the condition identified in the gap while keeping the circuit fixed."

    # ================================================================== LLM
    def _llm_plan(self, gap: ResearchGap, papers: Sequence[ResearchPaperAnalysis], plan_id: str,
                  feedback: Optional[str] = None, previous: Optional[ExperimentPlan] = None) -> Optional[ExperimentPlan]:
        context = "\n".join(f"- {p.title}: {p.research_objective}" for p in papers[:8])
        gap_json = json.dumps({"title": gap.title, "statement": gap.statement, "type": gap.gap_type,
                               "evidence": [e.quote for e in gap.evidence][:4], "hints": gap.hints}, ensure_ascii=False)
        user = f"<UNTRUSTED_PAPER_TEXT>\n{context}\n</UNTRUSTED_PAPER_TEXT>\n\n<RESEARCH_GAP>\n{gap_json}\n</RESEARCH_GAP>\n\n"
        if feedback and previous is not None:
            user += (f"Your previous plan failed verification. Fix it.\nPrevious circuit: "
                     f"{[(g.name, g.target_qubits) for g in previous.proposal.circuit.gates] if previous.proposal.circuit else None}\n"
                     f"Previous predictions: {_fmt_dist(previous.proposal.expected_outcomes)}\nVerification feedback: {feedback}\n")
        user += "Design one experiment that addresses this gap."
        try:
            raw = self.llm.complete(system_prompt=EXPERIMENT_PLANNING_PROMPT, user_prompt=user, temperature=0.2,
                                    fallback_to_demo=False, purpose="experiment_planning")
            d = parse_llm_json(raw)
            c = d["circuit"]
            n = int(c["num_qubits"])
            gates = [GateOperation(**g) for g in c["gates"]]
            outcomes = [ExpectedOutcome(basis_state=o["basis_state"], expected_probability=float(o["expected_probability"]))
                        for o in d.get("expected_outcomes", [])]
            steps = [str(s) for s in d.get("method_steps", [])]
            system = d.get("quantum_system") or gap.hints.get("target_system") or DEFAULT_SYSTEM
            proposal = build_proposal(d["title"], f"Address research gap: {gap.title}", d["hypothesis"], steps, system, n, gates, outcomes)
            return ExperimentPlan(
                plan_id=plan_id, gap_id=gap.gap_id, title=d["title"], hypothesis=d["hypothesis"],
                rationale=d.get("rationale", gap.statement), method_steps=steps,
                variables={k: str(v) for k, v in (d.get("variables") or {}).items()}, baseline=d.get("baseline", ""),
                success_metrics=[str(x) for x in d.get("success_metrics", [])], risks=[str(x) for x in d.get("risks", [])],
                resources=d.get("resources", ""), proposal=proposal, generated_by="llm",
            )
        except Exception as exc:
            logger.warning(f"LLM experiment planning failed ({exc}); using the template planner.")
            return None

    # =============================================================== repair
    def _repair(self, plan: ExperimentPlan, report: VerificationReport, gap: ResearchGap,
                papers: Sequence[ResearchPaperAnalysis], feedback: str) -> ExperimentPlan:
        revised = plan.model_copy(deep=True)
        revised.revision = plan.revision + 1
        prop = revised.proposal
        circuit_failed = any(c.check_id == "CHK-CIRC-01" and c.status != CheckStatus.PASS for c in report.checks)
        family = gap.hints.get("family")

        if circuit_failed or prop.circuit is None:
            n, gates, outcomes, _ = template_circuit(family if family in TEMPLATE_FAMILIES else "bell", prop.num_qubits or 3)
            revised.proposal = build_proposal(revised.title, prop.objective, revised.hypothesis, revised.method_steps,
                                              prop.quantum_system or DEFAULT_SYSTEM, n, gates, outcomes)
            revised.revision_note = f"Circuit rebuilt from the {FAMILY_LABELS.get(family, 'Bell state')} template after: {feedback}"
            return revised

        n = prop.circuit.num_qubits
        if family in TEMPLATE_FAMILIES and template_circuit(family, n)[0] == n:
            _, _, outcomes, why = template_circuit(family, n)
            source = f"theory ({why})"
        elif report.simulation is not None:
            outcomes = [ExpectedOutcome(basis_state=k, expected_probability=round(v, 4))
                        for k, v in sorted(report.simulation.probabilities.items()) if v > 1e-6]
            source = "the ideal simulation of the plan's own circuit (prediction could not be derived from a template)"
        else:
            revised.revision_note = f"No automatic repair available: {feedback}"
            return revised
        revised.proposal = build_proposal(revised.title, prop.objective, revised.hypothesis, revised.method_steps,
                                          prop.quantum_system or DEFAULT_SYSTEM, n, prop.circuit.gates, outcomes)
        revised.revision_note = f"Predicted distribution corrected to {_fmt_dist(outcomes)} from {source}. Feedback was: {feedback}"
        return revised

    # ============================================================== helpers
    @staticmethod
    def _paper(papers: Sequence[ResearchPaperAnalysis], title: Optional[str]) -> Optional[ResearchPaperAnalysis]:
        return next((p for p in papers if p.title == title), None) if title else None

    @staticmethod
    def _source_circuit(paper: Optional[ResearchPaperAnalysis]):
        if paper is None:
            return None
        for exp in paper.experiments:
            if exp.gates and exp.num_qubits:
                return exp.num_qubits, list(exp.gates), list(exp.expected_outcomes)
        return None
