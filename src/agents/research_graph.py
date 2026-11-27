"""
Orchestrated research run (Phase 7): LangGraph, used only where it adds reliability.

    collect -> analyse -> gaps -> [researcher: select a gap] -> novelty (re-query <= 2 while coverage is low)
      -> plan -> [researcher: edit / approve in the Plans step] -> verify (revise <= 1, then the researcher
      approves again) -> report

- Every step is a plain function from the earlier phases; the graph adds order, bounded loops, human pauses,
  SQLite checkpoints and a timed execution log. No node exchanges free text with another model.
- State is checkpointed after every node in a SQLite file, keyed by the run's thread id, so a run continues
  after the app restarts. A node that raises leaves the last checkpoint intact: "Retry" re-runs that node.
- The researcher's decisions are never taken by the graph: it waits (interrupt) for a gap choice and for the
  plan decision made in the Plans step, and reads the decision from the database.
"""

import operator
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select

from ..experiment_planning import ProposalPlanner, apply_edit, load_plan, save_draft
from ..experiment_planning.proposal_planner import revise_after_verification
from ..gap_discovery import GapGenerator, load_assessment, load_gap, save_gaps, score_gap, weighted_total
from ..storage import Publication, ReportRecord, ResearchRunRecord, session_scope
from ..verification.plan_verifier import verify_plan

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = ["collect", "analyse", "gaps", "select_gap", "novelty", "plan", "review", "verify", "revise", "report"]
NODE_LABELS = {"collect": "Collect literature", "analyse": "Analyse papers", "gaps": "Find gaps", "select_gap": "You choose a gap",
               "novelty": "Check novelty", "plan": "Draft a plan", "review": "You review the plan", "verify": "Verify by simulation",
               "revise": "Revise once after a failure", "report": "Summarise the run"}
MAX_NOVELTY_RETRIES, MAX_REVISIONS = 2, 1


def checkpoint_path() -> str:
    return os.getenv("RESEARCH_CHECKPOINTS") or os.path.join(ROOT, "data", "checkpoints.db")


class RunState(TypedDict, total=False):
    params: Dict[str, Any]
    log: Annotated[List[Dict[str, Any]], operator.add]
    candidates: List[Dict[str, Any]]
    gap_id: Optional[int]
    novelty: Dict[str, Any]
    novelty_attempts: int
    lineage: Optional[int]
    plan_status: Optional[str]
    verdict: Optional[str]
    failed_checks: List[str]
    revisions: int
    report_id: Optional[int]
    outcome: str


@dataclass
class Services:
    llm: Any
    ingestor: Any
    engine: Any
    analyzer: Any
    checker: Any
    embed: Optional[Callable] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def logged(name: str):
    """Times the node and appends one execution-log entry. Interrupts and errors pass through untouched."""
    def deco(fn):
        def run(state: RunState) -> Dict[str, Any]:
            started, t0 = _now(), time.perf_counter()
            out = fn(state) or {}
            entry = {"node": name, "label": NODE_LABELS[name], "started": started,
                     "ms": round((time.perf_counter() - t0) * 1000), "summary": out.pop("_summary", ""),
                     "calls": out.pop("_calls", [])}
            out["log"] = [entry]
            return out
        return run
    return deco


def build_graph(sv: Services, db_url: Optional[str] = None) -> StateGraph:
    P = lambda s: s.get("params") or {}

    @logged("collect")
    def collect(s):
        p = P(s)
        if not p.get("collect") or not (p.get("query") or "").strip():
            return {"_summary": "Skipped: the run uses the papers already in your corpus."}
        rep = sv.ingestor.collect(p["query"], sources=p.get("sources") or ("openalex", "arxiv", "crossref"),
                                  limit_per_source=int(p.get("limit") or 10), year_from=p.get("year_from"))
        calls = [f"{o.source}: {o.status}, {o.returned} returned, {o.new} new" for o in rep.outcomes]
        return {"_summary": f"“{p['query']}”: {len(set(rep.publication_ids))} publications ({rep.new} new).", "_calls": calls}

    @logged("analyse")
    def analyse(s):
        n = int(P(s).get("analyse_n") or 0)
        with session_scope(db_url) as ses:
            todo = [x.id for x in ses.scalars(select(Publication).where(
                Publication.analysed_at.is_(None), Publication.curation_status != "excluded",
                Publication.is_synthetic.is_(False), Publication.discovered_via == "collection",
                Publication.abstract.is_not(None)).order_by(Publication.id).limit(n))] if n else []
        calls = []
        for pid in todo:
            rec = sv.analyzer.analyse(pid)
            calls.append(f"PUB-{pid}: {len(rec.statements)} statements ({rec.llm or 'rule-based'})")
        return {"_summary": f"Analysed {len(todo)} paper{'s' if len(todo) != 1 else ''}." if n else
                "Skipped: no new papers requested; uses papers already analysed.", "_calls": calls}

    @logged("gaps")
    def gaps(s):
        found = save_gaps(GapGenerator(db_url, embed=sv.embed, llm=sv.llm).generate(), db_url)
        ranked = []
        for g in found:
            a = load_assessment(g.id, db_url)
            ranked.append({"id": g.id, "title": g.title, "origin": g.origin, "statement": g.statement,
                           "classification": a["classification"] if a else None,
                           "total": weighted_total(score_gap(g, a))})
        ranked.sort(key=lambda x: -x["total"])
        top = ranked[:int(P(s).get("top_gaps") or 8)]
        return {"candidates": top, "_summary": f"{len(found)} candidate gaps; the top {len(top)} are offered to you."}

    @logged("select_gap")
    def select_gap(s):
        preset = P(s).get("gap_id")
        cands = s.get("candidates") or []
        if preset:
            choice = int(preset)
        else:
            choice = interrupt({"kind": "select_gap", "candidates": cands,
                                "message": "Choose the gap this run should investigate."})
            choice = int(str(choice).replace("CG-", ""))
        if load_gap(choice, db_url) is None:
            raise ValueError(f"No gap CG-{choice}")
        return {"gap_id": choice, "_summary": f"CG-{choice} chosen" + (" (set when the run started)." if preset else " by you.")}

    @logged("novelty")
    def novelty(s):
        attempt = s.get("novelty_attempts") or 0
        base = sv.checker.per_query
        sv.checker.per_query = base * (attempt + 1)         # a re-query widens the search
        try:
            a = sv.checker.check(s["gap_id"])
        finally:
            sv.checker.per_query = base
        cov = a["coverage"]
        return {"novelty": {"classification": a["classification"], "label": a["label"], "confidence": a["confidence"],
                            "coverage_ok": cov["coverage_ok"], "reason": a["findings"]["reason"]},
                "novelty_attempts": attempt + 1,
                "_summary": f"{a['label']} (confidence {a['confidence']:.2f}).",
                "_calls": [f"{q['kind']}: “{q['query']}”" for q in cov["queries"]]
                + [f"{len(cov['sources_ok'])} of {len(cov['sources_attempted'])} sources answered; {cov['screened']} screened"]
                + ([f"attempt {attempt + 1}: {base * (attempt + 1)} results per query"] if attempt else [])}

    def after_novelty(s) -> str:
        n = s.get("novelty") or {}
        low = n.get("classification") == "insufficient_coverage" and not n.get("coverage_ok")
        return "novelty" if low and (s.get("novelty_attempts") or 0) <= MAX_NOVELTY_RETRIES else "plan"

    @logged("plan")
    def plan(s):
        p = ProposalPlanner(sv.llm, db_url).draft(s["gap_id"])
        lin = save_draft(p, db_url)
        return {"lineage": lin, "_summary": f"Drafted RP-{lin} ({p.generated_by}); {len(p.missing_information)} item(s) missing.",
                "_calls": [f"feasibility: {p.feasibility_class}"] + [f"warning: {w}" for w in p.warnings]}

    @logged("review")
    def review(s):
        lin = s["lineage"]
        interrupt({"kind": "review_plan", "lineage": lin, "plan_id": f"RP-{lin}",
                   "message": f"Open RP-{lin} in Plans, edit it if needed, then approve or reject it. Continue the run afterwards."})
        status = load_plan(lin, db_url)["status"]
        return {"plan_status": status, "_summary": {"approved": f"RP-{lin} approved by you.", "rejected": f"RP-{lin} rejected by you.",
                                                    "draft": f"RP-{lin} not decided yet; waiting again."}[status]}

    def after_review(s) -> str:
        return {"approved": "verify", "rejected": "report"}.get(s.get("plan_status"), "review")

    @logged("verify")
    def verify(s):
        res = verify_plan(s["lineage"], sv.engine, db_url)
        failed = [f"{c.check_id}: {c.error_message or c.details}"[:300] for c in res["report"].checks if c.status.value == "FAIL"]
        return {"verdict": res["verdict"], "failed_checks": failed,
                "_summary": f"Verdict on revision {res['revision']}: {res['verdict']}.", "_calls": failed[:5]}

    def after_verify(s) -> str:
        return "revise" if s.get("verdict") == "fail" and (s.get("revisions") or 0) < MAX_REVISIONS else "report"

    @logged("revise")
    def revise(s):
        lin = s["lineage"]
        circuit, note = revise_after_verification(sv.llm, load_plan(lin, db_url)["proposal"], s.get("failed_checks") or [])
        if circuit is None:
            return {"revisions": (s.get("revisions") or 0) + 1, "plan_status": "unrevised", "_summary": note}
        call = getattr(sv.llm, "last_call", {}) or {}
        plan_ = apply_edit(lin, {"circuit": circuit.model_dump(mode="json")}, note=f"Revised after verification: {note}",
                           editor=f"{call.get('provider', 'llm')}/{call.get('model', '?')}", db_url=db_url)
        return {"revisions": (s.get("revisions") or 0) + 1, "plan_status": "draft",
                "_summary": f"Revision {plan_['revision']} written; it needs your approval before it is verified again.",
                "_calls": [note]}

    def after_revise(s) -> str:
        return "review" if s.get("plan_status") == "draft" else "report"

    @logged("report")
    def report(s):
        if s.get("verdict"):
            outcome = {"pass": "Plan verified in simulation", "fail": "Plan failed verification",
                       "hold": "Plan could not be verified (missing information)"}[s["verdict"]]
        elif s.get("plan_status") == "rejected":
            outcome = "Plan rejected by you"
        elif not s.get("candidates") and not s.get("gap_id"):
            outcome = "No gaps found in the analysed corpus"
        else:
            outcome = "Run ended"
        payload = {"params": P(s), "gap_id": s.get("gap_id"), "novelty": s.get("novelty"), "lineage": s.get("lineage"),
                   "verdict": s.get("verdict"), "revisions": s.get("revisions") or 0, "outcome": outcome,
                   "log": [{k: e[k] for k in ("node", "started", "ms", "summary")} for e in s.get("log") or []]}
        with session_scope(db_url) as ses:
            r = ReportRecord(kind="research_investigation", formats=["json"], payload=payload)
            ses.add(r)
            ses.flush()
            rid = r.id
        return {"report_id": rid, "outcome": outcome, "_summary": outcome + "."}

    g = StateGraph(RunState)
    for name, fn in (("collect", collect), ("analyse", analyse), ("gaps", gaps), ("select_gap", select_gap),
                     ("novelty", novelty), ("plan", plan), ("review", review), ("verify", verify), ("revise", revise),
                     ("report", report)):
        g.add_node(name, fn)
    g.add_edge(START, "collect")
    g.add_edge("collect", "analyse")
    g.add_edge("analyse", "gaps")
    g.add_conditional_edges("gaps", lambda s: "select_gap" if s.get("candidates") or P(s).get("gap_id") else "report",
                            ["select_gap", "report"])
    g.add_edge("select_gap", "novelty")
    g.add_conditional_edges("novelty", after_novelty, ["novelty", "plan"])
    g.add_edge("plan", "review")
    g.add_conditional_edges("review", after_review, ["verify", "report", "review"])
    g.add_conditional_edges("verify", after_verify, ["revise", "report"])
    g.add_conditional_edges("revise", after_revise, ["review", "report"])
    g.add_edge("report", END)
    return g


class ResearchRunner:
    """Starts, resumes and retries runs; keeps the research_runs index in step with the checkpoint."""

    def __init__(self, services: Services, db_url: Optional[str] = None, checkpoints: Optional[str] = None):
        self.db_url = db_url
        path = checkpoints or checkpoint_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        self.graph = build_graph(services, db_url).compile(checkpointer=self.saver)

    @staticmethod
    def _cfg(thread: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread}}

    def start(self, params: Dict[str, Any]) -> int:
        thread = uuid.uuid4().hex
        with session_scope(self.db_url) as s:
            r = ResearchRunRecord(thread_id=thread, params=params, status="running", current="collect")
            s.add(r)
            s.flush()
            rid = r.id
        self._advance(rid, {"params": params, "revisions": 0, "novelty_attempts": 0})
        return rid

    def resume(self, run_id: int, value: Any) -> None:
        rec = self._record(run_id)
        if rec["status"] != "waiting":
            raise ValueError("This run is not waiting for you.")
        self._advance(run_id, Command(resume=value))

    def retry(self, run_id: int) -> None:
        if self._record(run_id)["status"] != "failed":
            raise ValueError("Only a failed run can be retried.")
        self._advance(run_id, None)                         # re-runs the node that failed, from its checkpoint

    def cancel(self, run_id: int) -> None:
        with session_scope(self.db_url) as s:
            r = s.get(ResearchRunRecord, run_id)
            r.status, r.waiting, r.updated_at = "cancelled", None, datetime.now(timezone.utc)

    def _record(self, run_id: int) -> Dict[str, Any]:
        with session_scope(self.db_url) as s:
            r = s.get(ResearchRunRecord, run_id)
            if r is None:
                raise ValueError(f"No run {run_id}")
            return {"thread": r.thread_id, "status": r.status}

    def _advance(self, run_id: int, inp: Any) -> None:
        thread = self._record(run_id)["thread"]
        with session_scope(self.db_url) as s:
            r = s.get(ResearchRunRecord, run_id)
            r.status, r.waiting, r.error, r.updated_at = "running", None, None, datetime.now(timezone.utc)
        error = None
        try:
            self.graph.invoke(inp, self._cfg(thread))
        except Exception as exc:                           # the checkpoint before the failing node is intact
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        st = self.graph.get_state(self._cfg(thread))
        nxt = st.next[0] if st.next else None
        waits = [i.value for t in st.tasks for i in (t.interrupts or [])]
        with session_scope(self.db_url) as s:
            r = s.get(ResearchRunRecord, run_id)
            r.current, r.updated_at = nxt, datetime.now(timezone.utc)
            if error:
                r.status, r.error = "failed", error
                r.failures = list(r.failures or []) + [{"node": nxt, "error": error, "at": _now()}]
            elif waits:
                r.status, r.waiting = "waiting", waits[0]
            elif nxt is None:
                r.status = "done"

    def status(self, run_id: int) -> Dict[str, Any]:
        with session_scope(self.db_url) as s:
            r = s.get(ResearchRunRecord, run_id)
            rec = {"id": r.id, "thread": r.thread_id, "params": r.params, "status": r.status, "current": r.current,
                   "waiting": r.waiting, "error": r.error, "failures": r.failures or [],
                   "created_at": r.created_at.isoformat() if r.created_at else None,
                   "updated_at": r.updated_at.isoformat() if r.updated_at else None}
        vals = self.graph.get_state(self._cfg(rec["thread"])).values or {}
        rec.update({k: vals.get(k) for k in ("candidates", "gap_id", "novelty", "lineage", "plan_status", "verdict",
                                             "revisions", "report_id", "outcome")})
        rec["log"] = vals.get("log") or []
        return rec

    def list_runs(self) -> List[Dict[str, Any]]:
        with session_scope(self.db_url) as s:
            ids = [r.id for r in s.scalars(select(ResearchRunRecord).order_by(ResearchRunRecord.id.desc()))]
        return [self.status(i) for i in ids]
