"""
Plans, edits and approvals in the research database (auditable).

A plan is a lineage of revisions. Revision 0 is the generated draft; every researcher edit stores a new revision
with the fields it changed. Only the researcher approves or rejects, and each decision is an Approval row on the
exact revision it applies to. Editing an approved plan creates a new draft revision: approval never carries over.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import ValidationError
from sqlalchemy import select

from ..storage import Approval, ExperimentPlanRecord, session_scope
from .proposal_planner import missing
from .schema import EDITABLE_LIST, EDITABLE_TEXT, REQUIRED, SECTIONS, CircuitDraft, Parameter, ResearchProposal


class PlanEditError(ValueError):
    pass


def save_draft(p: ResearchProposal, db_url: Optional[str] = None) -> int:
    """Store a generated draft as revision 0 of a new lineage; returns the lineage id."""
    with session_scope(db_url) as s:
        r = ExperimentPlanRecord(gap_id=p.gap_id, revision=0, status="draft", feasibility_class=p.feasibility_class,
                                 generator=p.generated_by, plan={"proposal": p.model_dump(mode="json"), "changed": [], "note": None})
        s.add(r)
        s.flush()
        r.plan = {**r.plan, "lineage": r.id}
        return r.id


def _rows(s, lineage: int) -> List[ExperimentPlanRecord]:
    rows = [r for r in s.scalars(select(ExperimentPlanRecord)) if (r.plan or {}).get("lineage") == lineage]
    return sorted(rows, key=lambda r: r.revision)


def apply_edit(lineage: int, changes: Dict[str, Any], note: str = "", editor: str = "researcher",
               db_url: Optional[str] = None) -> Dict[str, Any]:
    """Validate the researcher's changes and store them as the next revision (status draft)."""
    with session_scope(db_url) as s:
        rows = _rows(s, lineage)
        if not rows:
            raise PlanEditError(f"No plan {lineage}")
        cur = ResearchProposal(**rows[-1].plan["proposal"])
        new = cur.model_copy(deep=True)
        changed: List[str] = []
        for k, v in changes.items():
            if k in EDITABLE_TEXT:
                v = str(v or "").strip()
                if v != getattr(cur, k):
                    setattr(new, k, v)
                    changed.append(k)
            elif k in EDITABLE_LIST:
                v = [x.strip() for x in (v.splitlines() if isinstance(v, str) else v or []) if str(x).strip()]
                if v != getattr(cur, k):
                    setattr(new, k, v)
                    changed.append(k)
            elif k == "parameters":
                old = {(p.name, p.value, p.unit): p for p in cur.parameters}
                params = []
                for prm in v or []:
                    name, value, unit = (str(prm.get(x) or "").strip() for x in ("name", "value", "unit"))
                    if not name and not value:
                        continue
                    if not name or not value:
                        raise PlanEditError("Every parameter needs a name and a value.")
                    same = old.get((name, value, unit))
                    params.append(same or Parameter(name=name, value=value, unit=unit, source="researcher",
                                                    basis=str(prm.get("basis") or "Set by the researcher.")))
                if [p.model_dump() for p in params] != [p.model_dump() for p in cur.parameters]:
                    new.parameters = params
                    changed.append("parameters")
            elif k == "circuit":
                try:
                    c = CircuitDraft(**v) if v else None
                except (ValidationError, TypeError) as exc:
                    raise PlanEditError(f"The circuit is not valid: {str(exc)[:200]}")
                if (c.model_dump() if c else None) != (cur.circuit.model_dump() if cur.circuit else None):
                    new.circuit = c
                    changed.append("circuit")
            elif k == "feasibility_class":
                if v not in ("simulatable", "needs_hardware", "needs_physical_validation"):
                    raise PlanEditError("Unknown feasibility class.")
                if v != cur.feasibility_class:
                    new.feasibility_class, new.feasibility_reason = v, f"Set by the researcher (was {cur.feasibility_class})."
                    changed.append("feasibility_class")
        if not changed:
            return load_plan(lineage, db_url)
        new.missing_information = [m for m in cur.missing_information if not m.endswith("(not written yet)")] + missing(new)
        if cur.generated_by.endswith("+ researcher edits") or cur.generated_by == "researcher":
            new.generated_by = cur.generated_by
        else:
            new.generated_by = f"{cur.generated_by} + researcher edits"
        r = ExperimentPlanRecord(gap_id=cur.gap_id, revision=rows[-1].revision + 1, status="draft",
                                 feasibility_class=new.feasibility_class, generator=editor,
                                 plan={"proposal": new.model_dump(mode="json"), "lineage": lineage, "changed": changed,
                                       "note": note.strip() or None})
        s.add(r)
    return load_plan(lineage, db_url)


def decide(lineage: int, decision: str, note: str = "", reviewer: Optional[str] = None,
           db_url: Optional[str] = None) -> Dict[str, Any]:
    """Record the researcher's decision on the latest revision. Approval needs every required section."""
    if decision not in ("approved", "rejected", "changes_requested"):
        raise PlanEditError("Unknown decision.")
    with session_scope(db_url) as s:
        rows = _rows(s, lineage)
        if not rows:
            raise PlanEditError(f"No plan {lineage}")
        latest = rows[-1]
        p = ResearchProposal(**latest.plan["proposal"])
        if decision == "approved":
            empty = [lbl for f, lbl, _ in SECTIONS if f in REQUIRED and not getattr(p, f)]
            if empty:
                raise PlanEditError("Fill in before approving: " + ", ".join(empty) + ".")
        if decision in ("rejected", "changes_requested") and len(note.strip()) < 10:
            raise PlanEditError("Add a short note (10+ characters) saying why.")
        latest.status = {"approved": "approved", "rejected": "rejected", "changes_requested": "draft"}[decision]
        s.add(Approval(object_type="plan", object_id=latest.id, decision=decision, reviewer=reviewer, note=note.strip() or None,
                       decided_at=datetime.now(timezone.utc)))
    return load_plan(lineage, db_url)


def load_plan(lineage: int, db_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with session_scope(db_url) as s:
        rows = _rows(s, lineage)
        if not rows:
            return None
        ids = [r.id for r in rows]
        approvals = list(s.scalars(select(Approval).where(Approval.object_type == "plan", Approval.object_id.in_(ids))
                                   .order_by(Approval.decided_at)))
        rev_of = {r.id: r.revision for r in rows}
        latest = rows[-1]
        return {
            "lineage": lineage, "record_id": latest.id, "revision": latest.revision, "status": latest.status,
            "gap_id": latest.gap_id, "proposal": ResearchProposal(**latest.plan["proposal"]),
            "history": [{"revision": r.revision, "by": r.generator, "changed": (r.plan or {}).get("changed", []),
                         "note": (r.plan or {}).get("note"), "at": r.created_at.isoformat() if r.created_at else None,
                         "status": r.status} for r in rows],
            "approvals": [{"revision": rev_of[a.object_id], "decision": a.decision, "note": a.note, "reviewer": a.reviewer,
                           "at": a.decided_at.isoformat() if a.decided_at else None} for a in approvals],
        }


def list_plans(db_url: Optional[str] = None) -> List[Dict[str, Any]]:
    with session_scope(db_url) as s:
        lineages = sorted({(r.plan or {}).get("lineage") for r in s.scalars(select(ExperimentPlanRecord))} - {None}, reverse=True)
    return [p for p in (load_plan(l, db_url) for l in lineages) if p]
