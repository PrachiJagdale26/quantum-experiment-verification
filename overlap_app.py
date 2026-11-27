"""
Overlap: LLM-Based Research Gap Identification and Experiment Planning for quantum literature.

    Papers  ->  Gaps  ->  Plans (each verified by Qiskit simulation + rule audit)

plus "Check a proposal" (verify a single experiment proposal) and the original
research dashboard with all of its tabs.

    streamlit run overlap_app.py
"""

import base64
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.document_processing.document_extractor import DocumentExtractor
from src.evaluation.benchmark import load_benchmark_cases
from src.knowledge_base.retriever import QuantumRetriever
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.llm.client import ConfigurableLLMClient
from src.llm.extractor import ProposalExtractor
from src.research import ExperimentPlanner, GapFinder, PaperIngestError, PaperLibrary, ResearchPipeline
from src.schemas.proposal import ProposalSpecification
from src.ui.overlap_view import build_gap_view, build_paper_view, build_plan_view, build_view, overlap
from src.verification.engine import VerificationEngine

SAMPLE_DOC = os.path.join(ROOT, "data", "sample_proposals", "sample_bell_state_proposal.pdf")
SUPPORTED = (".pdf", ".docx", ".txt")
DASHBOARD_PATH = "dashboard"

# The component's iframe is pinned to the browser window, so it never depends on
# Streamlit's own layout and always scrolls on its own.
REVIEW_CSS = """<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
footer, #MainMenu {display:none !important;}
[data-stale="true"], .stale-element {opacity:1 !important; transition:none !important;}
iframe[src*="overlap_view.overlap"] {position:fixed !important; inset:0 !important; width:100vw !important;
  height:100vh !important; z-index:999990; border:0; display:block;}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {overflow:hidden !important;}
</style>"""


@st.cache_resource(show_spinner=False)
def get_services():
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    probe = ConfigurableLLMClient(demo_mode=False)
    live, _ = probe.is_configured()
    llm = ConfigurableLLMClient(demo_mode=not live)
    engine = VerificationEngine(retriever=QuantumRetriever(kb), llm_client=llm)
    if live:
        names = {"vertex": "Vertex AI", "gemini": "Gemini", "openai": "OpenAI", "anthropic": "Anthropic"}
        mode = f"LLM: {names.get(llm.provider_name, llm.provider_name)}, {llm.model_name}"
    else:
        mode = "LLM: demo mode (rule-based)"
    return kb, llm, engine, ProposalExtractor(llm), DocumentExtractor(), mode


def research_page():
    st.markdown(REVIEW_CSS, unsafe_allow_html=True)
    kb, llm, engine, extractor, doc_extractor, mode = get_services()
    ss = st.session_state

    if "pipeline" not in ss:
        library = PaperLibrary(extractor, kb, doc_extractor)
        ss.pipeline = ResearchPipeline(library, GapFinder(llm, kb), ExperimentPlanner(llm), engine)
        ss.paper_errors = []      # failed uploads shown in the Papers list
        ss.plan_times = {}
        ss.seq = 0
        ss.done = []
        ss.focus = None
        ss.last_eid = None
        ss.queue = None           # single-proposal queue, loaded on first use
    pipe: ResearchPipeline = ss.pipeline

    def now_label() -> str:
        return datetime.now().strftime("Verified today, %H:%M")

    def focus(sec: str, item_id: Optional[str] = None, msg: Optional[str] = None):
        ss.focus = {"sec": sec, "id": item_id, "msg": msg, "seq": ss.seq}

    # ----------------------------------------------------- single proposals
    def next_id() -> str:
        ss.counter = ss.get("counter", 400) + 1
        return f"P-{ss.counter:04d}"

    def make_item(proposal: ProposalSpecification, label: str, raw_gates=None, token: Optional[str] = None) -> Dict[str, Any]:
        return {"id": next_id(), "proposal": proposal, "report": engine.verify_proposal(proposal),
                "label": label, "raw_gates": raw_gates, "when": now_label(), "token": token, "dec": False}

    def load_benchmarks():
        ss.queue = []
        for case in load_benchmark_cases():
            data = case["proposal"]
            try:
                proposal = ProposalSpecification(**data)
            except Exception as exc:
                ss.queue.append({"id": next_id(), "state": "bad", "title": data.get("title", case["id"]),
                                 "file": case["id"], "why": str(exc)})
                continue
            ss.queue.append(make_item(proposal, f"Benchmark {case['id']}", raw_gates=(data.get("circuit") or {}).get("gates")))

    def verify_document(name: str, payload: bytes, token: str):
        ext = os.path.splitext(name)[1].lower()
        title = os.path.splitext(name)[0].replace("_", " ").replace("-", " ")
        why = None if ext in SUPPORTED else f"{ext or 'Files without an extension'} aren't supported. Upload a PDF, DOCX or TXT file."
        if why is None:
            try:
                doc = doc_extractor.extract_from_bytes(payload, filename=name)
                if not doc.full_text.strip():
                    why = "The document has no readable text layer. Upload a DOCX or a text-based PDF."
                else:
                    proposal = extractor.extract(raw_text=doc.full_text, document_title=doc.sections.get("title", title),
                                                 extracted_doc=doc, input_mode="upload")
                    item = make_item(proposal, name, token=token)
                    ss.queue.insert(0, item)
                    focus("proposals", item["id"], f"Verified {name}")
                    return
            except Exception as exc:
                why = f"The document couldn't be parsed: {exc}"
        item = {"id": next_id(), "state": "bad", "title": title, "file": name, "why": why, "token": token}
        ss.queue.insert(0, item)
        focus("proposals", item["id"], f"Couldn't read {name}")

    # ---------------------------------------------------------- event loop
    def handle(ev: Dict[str, Any]):
        kind = ev.get("kind")
        ss.seq += 1
        if ev.get("token"):
            ss.done = (ss.done + [ev["token"]])[-20:]
        if ss.queue is None and kind in ("upload", "sample", "reload"):
            ss.queue = []
        find = lambda i: next((x for x in (ss.queue or []) if x["id"] == i), None)

        if kind == "upload_papers":
            first_id, n_ok = None, 0
            for f in ev.get("files", []):
                name = f.get("name", "paper")
                try:
                    if f.get("size", 0) > 25e6:
                        raise PaperIngestError("The file is larger than 25 MB.")
                    pid, _ = pipe.library.add_document(base64.b64decode(f.get("b64") or b""), name)
                    n_ok += 1
                except Exception as exc:  # PaperIngestError, or anything a corrupt file triggers
                    pid = f"ERR-{len(ss.paper_errors) + 1:02d}"
                    ss.paper_errors.append({"id": pid, "state": "bad", "title": name, "file": name, "why": str(exc)})
                first_id = first_id or pid
            focus("papers", first_id, f"Analysed {n_ok} paper{'s' if n_ok != 1 else ''}")
        elif kind == "sample_papers":
            ids = pipe.library.load_samples()
            focus("papers", ids[0] if ids else None, f"Loaded {len(ids)} sample papers")
        elif kind == "remove_paper":
            pipe.library.remove(ev.get("id"))
            ss.paper_errors = [e for e in ss.paper_errors if e["id"] != ev.get("id")]
        elif kind == "clear_papers":
            pipe.library.clear()
            ss.paper_errors = []
        elif kind == "find_gaps":
            gaps = pipe.find_gaps()
            focus("gaps", gaps[0].gap_id if gaps else None, f"Found {len(gaps)} research gaps")
        elif kind == "plan_gap" and pipe.gap(ev.get("gap_id")):
            plan = pipe.plan(ev["gap_id"])
            ss.plan_times[plan.plan_id] = now_label()
            focus("plans", plan.plan_id, f"{plan.plan_id} " + {"pass": "verified", "fail": "fails verification",
                                                               "hold": "can't be verified yet"}[plan.verification_status])
        elif kind == "reverify" and ev.get("id") in pipe.plans:
            pipe.verify(ev["id"])
            ss.plan_times[ev["id"]] = now_label()
        # single-proposal checking
        elif kind == "upload":
            verify_document(ev.get("name", "proposal"), base64.b64decode(ev.get("b64") or b""), ev.get("token"))
        elif kind == "sample":
            with open(SAMPLE_DOC, "rb") as f:
                verify_document(os.path.basename(SAMPLE_DOC), f.read(), ev.get("token"))
        elif kind == "decide" and find(ev.get("id")):
            find(ev["id"])["dec"] = bool(ev.get("dec"))
        elif kind == "rerun" and find(ev.get("id")) and "proposal" in find(ev["id"]):
            item = find(ev["id"])
            item["report"], item["when"] = engine.verify_proposal(item["proposal"]), now_label()
        elif kind == "remove":
            ss.queue = [x for x in ss.queue if x["id"] != ev.get("id")]
        elif kind == "clear":
            ss.queue = []
        elif kind == "reload":
            load_benchmarks()
        elif kind == "nav":
            st.switch_page(dashboard)

    event = ss.get("overlap")
    if event and event.get("eid") != ss.last_eid:
        ss.last_eid = event["eid"]
        handle(event)

    # Arriving from the full dashboard's top bar: open the section that was clicked.
    target = ss.pop("overlap_sec", None)
    if target:
        ss.seq += 1
        ss.focus = {"sec": target, "id": None, "msg": None, "seq": ss.seq, "initial": True}
    if ss.queue is None:
        load_benchmarks()

    # ------------------------------------------------------------- views
    def proposal_view(item: Dict[str, Any]) -> Dict[str, Any]:
        if item.get("state") == "bad":
            return {k: item.get(k) for k in ("id", "title", "file", "why", "token")} | {"state": "bad"}
        v = build_view(item["id"], item["proposal"], item["report"], item["label"], item.get("raw_gates"), item["when"])
        v.update(dec=item["dec"], token=item.get("token"))
        return v

    planned: Dict[str, str] = {}
    for plan in pipe.plans.values():
        planned.setdefault(plan.gap_id, plan.plan_id)
    gap_titles = {g.gap_id: g.title for g in pipe.gaps}

    papers = [build_paper_view(pid, p) for pid, p in pipe.library.papers()] + ss.paper_errors
    gaps = [build_gap_view(g, planned.get(g.gap_id)) for g in pipe.gaps]
    plans = [build_plan_view(plan, pipe.reports[pid], gap_titles.get(plan.gap_id, plan.gap_id), ss.plan_times.get(pid, ""))
             for pid, plan in reversed(list(pipe.plans.items())) if pid in pipe.reports]
    if llm.live_failure:
        mode = f"LLM: {llm.live_failure}. Using rule-based analysis."
    meta = {"mode": mode, "seq": ss.seq, "done": ss.done, "agenda": pipe.agenda_markdown() if pipe.gaps else ""}
    overlap(papers, gaps, plans, [proposal_view(x) for x in ss.queue], ss.focus, meta)


BAR_CSS = """<style>
.st-key-ovbar {border-bottom: 1px solid color-mix(in srgb, currentColor 14%, transparent); padding: 2px 0 10px; margin-bottom: 6px;}
.st-key-ovbar .ov-brand {margin: 0 14px 0 0;}
.st-key-ovbar button {padding: 6px 12px; border-radius: 10px; min-height: 40px;}
.st-key-ovbar button p {font-size: 14px;}
.st-key-ovbar button:hover {background: color-mix(in srgb, currentColor 8%, transparent);}
.st-key-ovnav_dash button:disabled {opacity: 1 !important; cursor: default; color: inherit !important;
  background: color-mix(in srgb, currentColor 10%, transparent);}
.st-key-ovnav_dash button:disabled p {color: inherit !important; font-weight: 500;}
.st-key-ovbar .ov-arrow {opacity: .35; font-size: 13px; margin: 0 -4px;}
</style>"""
BRAND_HTML = ('<div class="ov-brand"><svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="17" class="sph"/>'
              '<ellipse cx="20" cy="20" rx="17" ry="5.5" class="eq"/><line x1="20" y1="20" x2="29" y2="8" class="vec"/>'
              '<circle cx="29" cy="8" r="3" class="tip"/></svg><div><b>Overlap</b></div></div>')


def dashboard_bar():
    """Overlap's top bar on the full dashboard: same steps, same session, one click back."""
    ss = st.session_state
    pipe = ss.get("pipeline")
    counts = {"papers": len(pipe.library) if pipe else 0, "gaps": len(pipe.gaps) if pipe else 0,
              "plans": len(pipe.plans) if pipe else 0}
    st.markdown(BAR_CSS, unsafe_allow_html=True)
    with st.container(horizontal=True, key="ovbar", vertical_alignment="center", gap="small"):
        st.markdown(BRAND_HTML, unsafe_allow_html=True)
        steps = [("papers", "Papers"), ("gaps", "Gaps"), ("plans", "Plans")]
        for i, (sec, label) in enumerate(steps):
            n = counts[sec]
            if st.button(f"{label} ({n})" if n else label, key=f"ovnav_{sec}", type="tertiary"):
                ss.overlap_sec = sec
                st.switch_page(research)
            if i < len(steps) - 1:
                st.markdown('<span class="ov-arrow">→</span>', unsafe_allow_html=True)
        if st.button("Check a proposal", key="ovnav_proposals", type="tertiary"):
            ss.overlap_sec = "proposals"
            st.switch_page(research)
        st.button("Full dashboard", key="ovnav_dash", type="tertiary", disabled=True)


st.set_page_config(page_title="Overlap · Research gaps and experiment plans", page_icon="⚛️", layout="wide")
research = st.Page(research_page, title="Research gaps", url_path="research", default=True)
dashboard = st.Page(os.path.join(ROOT, "app.py"), title="Full dashboard", url_path=DASHBOARD_PATH)
page = st.navigation([research, dashboard], position="hidden")
if page.url_path == DASHBOARD_PATH:
    dashboard_bar()
page.run()
