"""
ResearchPipeline: the end-to-end framework.

    1. Literature  PaperLibrary      ingest + analyse + index papers
    2. Gaps        GapFinder         find and rank what's missing across them
    3. Plans       ExperimentPlanner design an experiment for a chosen gap
    4. Verify      VerificationEngine simulate + audit the plan's circuit; revise once on failure
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..schemas.research_gap import GAP_TYPE_LABELS, ExperimentPlan, ResearchGap
from ..schemas.verification import VerificationReport
from ..verification.engine import VerificationEngine
from ..verification.verdict import overall_verdict
from .gap_finder import GapFinder
from .library import PaperLibrary
from .planner import ExperimentPlanner

VERDICT_LABELS = {"pass": "Plan verified", "fail": "Plan fails verification", "hold": "Can't verify yet"}


class ResearchPipeline:
    def __init__(self, library: PaperLibrary, gap_finder: GapFinder, planner: ExperimentPlanner, engine: VerificationEngine):
        self.library = library
        self.gap_finder = gap_finder
        self.planner = planner
        self.engine = engine
        self.gaps: List[ResearchGap] = []
        self.plans: Dict[str, ExperimentPlan] = {}
        self.reports: Dict[str, VerificationReport] = {}
        self.history: Dict[str, List[Tuple[ExperimentPlan, VerificationReport]]] = {}
        self._plan_counter = 0

    # ------------------------------------------------------------------ gaps
    def find_gaps(self) -> List[ResearchGap]:
        found = self.gap_finder.find([p for _, p in self.library.papers()])
        self.gaps = found + [g for g in self.gaps if g.gap_id.startswith("CG-")]   # corpus gaps added for planning stay
        return found

    def gap(self, gap_id: str) -> Optional[ResearchGap]:
        return next((g for g in self.gaps if g.gap_id == gap_id), None)

    # ----------------------------------------------------------------- plans
    def plan(self, gap_id: str, verify: bool = True) -> ExperimentPlan:
        gap = self.gap(gap_id)
        if gap is None:
            raise KeyError(f"Unknown gap {gap_id}")
        self._plan_counter += 1
        plan = self.planner.plan(gap, [p for _, p in self.library.papers()], f"PLAN-{self._plan_counter:02d}")
        self.plans[plan.plan_id] = plan
        if verify:
            self.verify(plan.plan_id)
        return self.plans[plan.plan_id]

    def verify(self, plan_id: str, allow_revision: bool = True) -> Tuple[ExperimentPlan, VerificationReport]:
        """Verify a plan; if it fails or is inconclusive, revise once and verify again."""
        plan = self.plans[plan_id]
        report = self.engine.verify_proposal(plan.proposal)
        plan.verification_status = overall_verdict(report)
        self.history.setdefault(plan_id, []).append((plan.model_copy(deep=True), report))

        if plan.verification_status != "pass" and allow_revision and plan.revision == 0:
            gap = self.gap(plan.gap_id)
            if gap is not None:
                revised = self.planner.revise(plan, report, gap, [p for _, p in self.library.papers()])
                revised_report = self.engine.verify_proposal(revised.proposal)
                revised.verification_status = overall_verdict(revised_report)
                self.history[plan_id].append((revised.model_copy(deep=True), revised_report))
                plan, report = revised, revised_report

        self.plans[plan_id] = plan
        self.reports[plan_id] = report
        return plan, report

    def add_plan(self, plan: ExperimentPlan) -> None:
        """Register an externally built plan (used by tests and manual edits)."""
        self.plans[plan.plan_id] = plan

    # ---------------------------------------------------------------- report
    def agenda_markdown(self) -> str:
        papers = self.library.papers()
        lines = ["# Research agenda", "",
                 f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from {len(papers)} paper(s).", "",
                 "## Literature analysed", ""]
        lines += [f"- **{p.title}** ({p.source_filename or 'uploaded'})" for _, p in papers]
        lines += ["", "## Research gaps (ranked)", "",
                  "| # | Gap | Type | Priority | Novelty | Evidence | Feasibility |",
                  "|---|---|---|---|---|---|---|"]
        for g in self.gaps:
            lines.append(f"| {g.gap_id} | {g.title} | {GAP_TYPE_LABELS[g.gap_type]} | {g.priority:.2f} | "
                         f"{g.novelty:.2f} | {g.evidence_strength:.2f} | {g.feasibility:.2f} |")
        for g in self.gaps:
            lines += ["", f"### {g.gap_id}: {g.title}", "", g.statement, "", "Evidence:"]
            lines += [f"- “{e.quote}” ({e.paper_title}{f', p. {e.page_number}' if e.page_number else ''})" for e in g.evidence]
        lines += ["", "## Experiment plans", ""]
        if not self.plans:
            lines.append("No plans yet.")
        for plan in self.plans.values():
            verdict = VERDICT_LABELS.get(plan.verification_status or "", "Not verified")
            lines += [f"### {plan.plan_id}: {plan.title}", "",
                      f"- **Addresses:** {plan.gap_id}", f"- **Verification:** {verdict}" +
                      (f" (revision {plan.revision}: {plan.revision_note})" if plan.revision else ""),
                      f"- **Hypothesis:** {plan.hypothesis}", f"- **Baseline:** {plan.baseline}", "", "**Method**", ""]
            lines += [f"{i}. {s}" for i, s in enumerate(plan.method_steps, 1)]
            lines += ["", "**Success metrics**", ""] + [f"- {m}" for m in plan.success_metrics]
            lines += ["", "**Risks**", ""] + [f"- {r}" for r in plan.risks]
            if plan.proposal.circuit:
                gates = ", ".join(f"{g.name} {' '.join(map(str, g.target_qubits))}" for g in plan.proposal.circuit.gates)
                lines += ["", f"**Circuit** ({plan.proposal.circuit.num_qubits} qubits): {gates}"]
            lines.append("")
        lines += ["---", "Gaps and plans are generated from the analysed papers; verification covers ideal simulation only. "
                  "Expert review is required before running any plan on hardware."]
        return "\n".join(lines)
