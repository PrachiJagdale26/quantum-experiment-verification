"""Research proposals for selected gaps: drafting, researcher edits and approval (Phase 5)."""

from .plan_store import PlanEditError, apply_edit, decide, list_plans, load_plan, save_draft
from .proposal_planner import ProposalPlanner, classify_feasibility
from .schema import FEASIBILITY_LABELS, PARAM_SOURCE_LABELS, SECTIONS, ResearchProposal

__all__ = ["PlanEditError", "apply_edit", "decide", "list_plans", "load_plan", "save_draft", "ProposalPlanner",
           "classify_feasibility", "FEASIBILITY_LABELS", "PARAM_SOURCE_LABELS", "SECTIONS", "ResearchProposal"]
