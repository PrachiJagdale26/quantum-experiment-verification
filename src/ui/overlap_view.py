"""
"Overlap" reviewer interface: maps real VerificationReport objects into the view
model consumed by the bidirectional Streamlit component in src/ui/overlap/.

The component is plain HTML/JS speaking the Streamlit component protocol, so it
needs no npm build step.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

from src.reporting.generator import ReportGenerator
from src.research.domain import FAMILY_LABELS
from src.schemas.research_gap import GAP_TYPE_LABELS, ExperimentPlan, ResearchGap
from src.schemas.research_paper import ResearchPaperAnalysis
from src.schemas.proposal import ProposalSpecification
from src.schemas.verification import CheckStatus, VerificationReport
from src.verification.verdict import overall_verdict

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlap")
_overlap_component = components.declare_component("overlap", path=_COMPONENT_DIR)

CORE_CHECKS = ("CHK-STRUC-01", "CHK-CIRC-01", "CHK-SIM-01", "CHK-DIST-01")
CHECK_NAMES = {
    "CHK-STRUC-01": "Qubit count",
    "CHK-CIRC-01": "Circuit gates",
    "CHK-SIM-01": "Statevector normalization",
    "CHK-DIST-01": "Outcome distribution",
    "CHK-LIT-01": "Literature grounding",
    "CHK-PHYS-01": "Hardware feasibility",
}
STATUS = {
    CheckStatus.PASS: "pass",
    CheckStatus.FAIL: "fail",
    CheckStatus.INCONCLUSIVE: "hold",
    CheckStatus.NOT_RUN: "not",
}
MISSING = {
    "CHK-STRUC-01": ("n", "Qubit count", "a qubit count"),
    "CHK-CIRC-01": ("U", "Gate sequence", "a gate sequence"),
    "CHK-DIST-01": ("P", "Expected outcomes", "expected outcomes"),
}


def overlap(papers: List[Dict[str, Any]], gaps: List[Dict[str, Any]], plans: List[Dict[str, Any]],
            proposals: List[Dict[str, Any]], focus: Optional[Dict[str, Any]], meta: Dict[str, Any],
            literature: Optional[List[Dict[str, Any]]] = None, key: str = "overlap"):
    """Render the Literature -> Papers -> Gaps -> Plans app (plus single-proposal checking). Returns the latest UI event."""
    return _overlap_component(literature=literature or [], papers=papers, gaps=gaps, plans=plans, proposals=proposals,
                              focus=focus, meta=meta, key=key, default=None)


def clean_error(text: Optional[str]) -> str:
    """Strip Pydantic boilerplate from engine error strings."""
    if not text:
        return ""
    m = re.search(r"Value error, (.*?)(?: \[type=|$)", text, re.S)
    msg = m.group(1) if m else text
    msg = re.sub(r"^(Detected critical circuit specification error: )?(Circuit specification error: )?", "", msg)
    msg = re.sub(r"\s+For further information.*$", "", msg, flags=re.S)
    return " ".join(msg.split()).strip()


def _plain_detail(check) -> str:
    if check.check_id == "CHK-PHYS-01":
        first = check.details.split("IMPORTANT")[0].strip()
        return f"{first} Ideal simulation only: noise and decoherence aren't modelled."
    if check.status == CheckStatus.FAIL and check.error_message:
        return clean_error(check.error_message)
    return clean_error(check.details) if "Value error" in check.details else check.details


def _fmt(p: float) -> str:
    return f"{p:.2f}"


def _classical_fidelity(claim: Dict[str, float], sim: Dict[str, float]) -> float:
    keys = set(claim) | set(sim)
    bc = sum(math.sqrt(max(claim.get(k, 0.0), 0.0) * max(sim.get(k, 0.0), 0.0)) for k in keys)
    return round(min(1.0, bc ** 2), 4)


def verdict_of(report: VerificationReport) -> str:
    return overall_verdict(report)


def build_view(
    item_id: str,
    proposal: ProposalSpecification,
    report: VerificationReport,
    source_label: str,
    raw_gates: Optional[List[Dict[str, Any]]] = None,
    when: str = "",
) -> Dict[str, Any]:
    """Translate a proposal + its verification report into the component's view model."""
    verdict = verdict_of(report)
    by_id = {c.check_id: c for c in report.checks}
    comparisons = report.outcome_comparisons
    tol = comparisons[0].tolerance if comparisons else 0.05

    sim = report.simulation.probabilities if report.simulation else None
    phase = (
        {c.basis_state: c.phase_rad for c in report.simulation.statevector_components}
        if report.simulation else {}
    )
    claim = {o.basis_state: o.expected_probability for o in comparisons} or {
        o.basis_state: o.expected_probability for o in proposal.expected_outcomes
    }

    # Gates to draw: the validated circuit, or the raw (rejected) gate list.
    gates, n = None, proposal.num_qubits
    if proposal.circuit:
        gates = [[g.name.upper(), list(g.target_qubits)] for g in proposal.circuit.gates]
        n = proposal.circuit.num_qubits
    elif raw_gates:
        gates = [[str(g.get("name", "?")).upper(), list(g.get("target_qubits", []))] for g in raw_gates]

    bad, circ_msg = None, ""
    circ = by_id.get("CHK-CIRC-01")
    if circ is not None and circ.status == CheckStatus.FAIL:
        circ_msg = clean_error(circ.error_message or circ.details)
        raw_err = f"{circ.error_message or ''} {circ.details}"
        m = re.search(r"Gate #(\d+)", raw_err)
        m0 = re.search(r"gates\.(\d+)", raw_err)
        if m:
            bad = int(m.group(1)) - 1
        elif m0:
            bad = int(m0.group(1))

    # Headline: the finding, in one sentence.
    if verdict == "fail" and circ_msg:
        m = re.search(r"Gate #(\d+) \((\w+)\) targets qubit index (\d+), which is out of bounds for an allocated (\d+)-qubit", circ_msg)
        m2 = re.search(r"Parametric gate '(\w+)' requires", circ_msg)
        if m:
            k = int(m.group(4))
            head = f"Gate {m.group(1)} ({m.group(2)}) acts on qubit {m.group(3)}, but the register only has qubits 0–{k - 1}."
        elif m2:
            head = f"The {m2.group(1)} gate has no rotation angle, so the circuit can't be simulated."
        else:
            head = circ_msg
    elif verdict == "fail" and by_id.get("CHK-DIST-01") and by_id["CHK-DIST-01"].status == CheckStatus.FAIL:
        shown = comparisons[:3]
        claimed = " and ".join(f"P({c.basis_state}) = {_fmt(c.expected_probability)}" for c in shown)
        got = " and ".join(_fmt(c.computed_probability) for c in shown)
        head = f"Claimed {claimed}; the circuit gives {got}."
    elif verdict == "fail":
        first = next(c for c in report.checks if c.status == CheckStatus.FAIL)
        head = _plain_detail(first)
    elif verdict == "pass":
        shown = ", ".join(f"P({c.basis_state}) = {_fmt(c.computed_probability)}" for c in comparisons[:3])
        more = f" and {len(comparisons) - 3} more" if len(comparisons) > 3 else ""
        head = f"Simulation matches every claimed outcome: {shown}{more}."
    else:
        gaps = [MISSING[cid][2] for cid in MISSING if by_id.get(cid) and by_id[cid].status == CheckStatus.INCONCLUSIVE]
        if gaps:
            phrase = gaps[0] if len(gaps) == 1 else ", ".join(gaps[:-1]) + " or " + gaps[-1]
            head = f"The proposal doesn't state {phrase}."
        else:
            first = next((c for c in report.checks if c.check_id in CORE_CHECKS and c.status != CheckStatus.PASS), None)
            head = _plain_detail(first) if first else "Verification couldn't reach a verdict."

    missing = [
        [MISSING[cid][0], MISSING[cid][1], by_id[cid].details]
        for cid in MISSING
        if by_id.get(cid) and by_id[cid].status == CheckStatus.INCONCLUSIVE
    ]

    checks = [
        [c.check_id, CHECK_NAMES.get(c.check_id, c.name), STATUS[c.status], _plain_detail(c)]
        for c in report.checks
    ]
    claims = [[c.claim_text, c.literature_support_status, c.computational_status] for c in report.claims_analysis]
    cites = [
        [c.source_title, c.page_number, round(c.similarity_score, 2), " ".join(c.passage.split())[:260]]
        for c in report.citations[:4]
    ]

    fields = [
        ("Document type", "Research paper" if proposal.document_type == "research_paper" else "Experiment proposal"),
        ("Source", source_label),
        ("Objective", proposal.objective),
        ("Hypothesis", proposal.hypothesis),
        ("Quantum system", proposal.quantum_system),
        ("Qubits", proposal.num_qubits),
        ("Initial state", proposal.initial_state or (proposal.circuit.initial_state if proposal.circuit else None)),
        ("Measurement basis", proposal.measurement_basis or (proposal.circuit.measurement_basis if proposal.circuit else None)),
        ("DOI", proposal.doi),
    ]
    warnings = list(proposal.extraction_warnings) + [clean_error(w) for w in proposal.potential_inconsistencies]
    spec = {
        "fields": [[k, str(v)] for k, v in fields if v not in (None, "")],
        "lists": [
            ["Gate sequence", [f"{g[0]} on qubit{'s' if len(g[1]) > 1 else ''} {', '.join(map(str, g[1]))}" for g in (gates or [])]],
            ["Expected outcomes", [f"P({o.basis_state}) = {o.expected_probability:.2f}, tolerance ±{o.tolerance}" for o in proposal.expected_outcomes]],
            ["Assumptions", list(proposal.assumptions)],
            ["Extraction warnings", list(dict.fromkeys(w for w in warnings if w))],
        ],
    }

    return {
        "id": item_id,
        "spec": spec,
        "title": proposal.title or "Untitled proposal",
        "file": source_label,
        "v": verdict,
        "head": head,
        "when": when,
        "n": n,
        "gates": gates,
        "bad": bad,
        "sim": sim,
        "phase": phase,
        "claim": claim,
        "tol": tol,
        "tvd": report.total_variation_distance,
        "fid": _classical_fidelity(claim, sim) if sim and claim else None,
        "missing": missing,
        "checks": checks,
        "claims": claims,
        "cites": cites,
        "summary": report.overall_summary,
        "report_id": report.report_id,
        "json": ReportGenerator.to_json(report),
        "md": ReportGenerator.to_markdown(report),
    }


# ----------------------------------------------------------------------------------
# Research-gap pipeline views (Papers -> Gaps -> Plans)
# ----------------------------------------------------------------------------------

def _trim(text: Optional[str], limit: int = 280) -> str:
    """Shorten to whole sentences (or whole words) so leads never end mid-word."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0] + "…"


def build_paper_view(paper_id: str, paper: ResearchPaperAnalysis) -> Dict[str, Any]:
    rule_based = any("rule-based" in w.lower() for w in paper.extraction_warnings)
    experiments = []
    for e in paper.experiments:
        experiments.append({
            "title": e.title,
            "system": e.quantum_system,
            "n": e.num_qubits,
            "gates": [[g.name.upper(), list(g.target_qubits)] for g in e.gates] or None,
            "outcomes": ", ".join(f"P({o.basis_state}) = {o.expected_probability:.2f}" for o in e.expected_outcomes),
            "findings": e.findings,
        })
    return {
        "id": paper_id,
        "title": paper.title,
        "file": paper.source_filename or "uploaded",
        "synthetic": "synthetic sample paper" in (paper.raw_text or "").lower(),
        "analysed_by": "Rule-based analysis" if rule_based else "LLM analysis",
        "doi": paper.doi,
        "objective": _trim(paper.research_objective),
        "problem": paper.research_problem,
        "systems": paper.quantum_systems_discussed,
        "frameworks": paper.algorithms_and_frameworks,
        "contributions": paper.key_contributions or paper.results_and_findings,
        "limitations": paper.limitations,
        "future_work": paper.future_work,
        "open_questions": paper.open_questions,
        "claims": paper.key_claims,
        "experiments": experiments,
        "signals": len(paper.limitations) + len(paper.future_work) + len(paper.open_questions),
    }


def build_gap_view(gap: ResearchGap, plan_id: Optional[str] = None) -> Dict[str, Any]:
    fam = gap.hints.get("family")
    return {
        "id": gap.gap_id,
        "title": gap.title,
        "statement": gap.statement,
        "type": gap.gap_type,
        "type_label": GAP_TYPE_LABELS[gap.gap_type],
        "priority": gap.priority,
        "novelty": gap.novelty,
        "feasibility": gap.feasibility,
        "evidence_strength": gap.evidence_strength,
        "testable": gap.testable_on_simulator,
        "family": FAMILY_LABELS.get(fam) if fam else None,
        "target": gap.hints.get("target_system"),
        "papers": len({e.paper_title for e in gap.evidence}),
        "evidence": [{"paper": e.paper_title, "file": e.source_filename, "page": e.page_number, "quote": e.quote}
                     for e in gap.evidence],
        "plan_id": plan_id,
    }


def build_plan_view(plan: ExperimentPlan, report: VerificationReport, gap_title: str, when: str = "") -> Dict[str, Any]:
    v = build_view(plan.plan_id, plan.proposal, report, f"Addresses {plan.gap_id}", None, when)
    v["head"] = v["head"].replace("every claimed outcome", "every predicted outcome").replace("Claimed ", "Predicted ")
    v.update({
        "kind": "plan",
        "title": plan.title,
        "gap_id": plan.gap_id,
        "gap_title": gap_title,
        "revision": plan.revision,
        "revision_note": plan.revision_note,
        "generated_by": plan.generated_by,
        "plan": {
            "hypothesis": plan.hypothesis,
            "rationale": plan.rationale,
            "method": plan.method_steps,
            "variables": [[k.capitalize(), val] for k, val in plan.variables.items()],
            "baseline": plan.baseline,
            "metrics": plan.success_metrics,
            "risks": plan.risks,
            "resources": plan.resources,
        },
    })
    return v


# ----------------------------------------------------------------------------------
# Literature Explorer views (Module 1)
# ----------------------------------------------------------------------------------
def build_publication_view(pub) -> Dict[str, Any]:
    """A stored Publication, with its provenance and availability stated explicitly."""
    meta = pub.raw_metadata or {}
    by_source = meta.get("by_source", {})
    availability = "full_text" if pub.full_text_available else ("abstract" if pub.abstract else "metadata_only")
    authors = list(pub.authors or [])
    return {
        "id": f"PUB-{pub.id}",
        "db_id": pub.id,
        "title": pub.title,
        "year": pub.year,
        "date": pub.publication_date,
        "venue": pub.venue,
        "authors": authors[:12],
        "author_count": len(authors),
        "doi": pub.doi,
        "arxiv_id": pub.arxiv_id,
        "openalex_id": pub.openalex_id,
        "abstract": pub.abstract,
        "abstract_source": pub.abstract_source,
        "availability": availability,
        "is_open_access": bool(pub.is_open_access),
        "source_url": pub.source_url,
        "pdf_url": pub.pdf_url,
        "sources": [{"name": name, "url": info.get("source_url"), "retrieved_at": (info.get("retrieved_at") or "")[:16].replace("T", " "),
                     "cited_by": info.get("cited_by_count"), "type": info.get("type")} for name, info in by_source.items()],
        "versions": [{"label": v.version_label, "date": v.version_date, "url": v.source_url}
                     for v in sorted(pub.versions, key=lambda v: v.version_date or "")],
        "references": len(meta.get("referenced_works", [])),
        "cited_by": max([i.get("cited_by_count") or 0 for i in by_source.values()] or [0]) or None,
        "conflicts": meta.get("identifier_conflicts", []),
        "fulltext_pages": meta.get("fulltext_pages"),
        "curation": pub.curation_status or "unreviewed",
        "curation_note": pub.curation_note,
        "analysed": pub.analysed_at is not None,
        "analysis_summary": meta.get("analysis"),
        # Work type as the sources report it (article, preprint, posted-content, ...), so a reader can tell
        # peer-reviewed articles from repository deposits; nothing is filtered out on the system's judgement.
        "work_types": sorted({str(i.get("type")).replace("-", " ") for i in by_source.values() if i.get("type")}),
        # No DOI other than a repository DOI (arXiv 10.48550, Zenodo 10.5281) -> no journal/conference version recorded.
        "preprint_only": not (pub.doi and not pub.doi.startswith(("10.48550/", "10.5281/"))),
    }


def build_search_runs_view(runs) -> List[Dict[str, Any]]:
    return [{"id": r.id, "source": r.source, "query": r.query, "status": r.status, "returned": r.result_count,
             "total": r.total_available, "error": r.error, "at": r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else ""}
            for r in runs]


# ----------------------------------------------------------------------------------
# Research analysis views (Phase 3)
# ----------------------------------------------------------------------------------
def build_analysis_view(record) -> Optional[Dict[str, Any]]:
    """A stored PaperAnalysisRecord, grouped by field, with each statement's provenance stated."""
    if record is None:
        return None
    from src.research_analysis import FIELD_LABELS
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for st in record.statements:
        p = st.provenance
        groups.setdefault(st.field, []).append({
            "text": st.text, "kind": st.kind, "confidence": round(st.confidence, 2), "extractor": st.extractor,
            "excerpt": p.excerpt if p else None, "verified": bool(p and p.verified), "page": p.page if p else None,
        })
    order = [f for f in FIELD_LABELS if f in groups]
    d = record.dimensions
    return {
        "from": record.analysed_from, "llm": record.llm, "warnings": record.warnings,
        "fields": [{"field": f, "label": FIELD_LABELS[f], "items": groups[f]} for f in order],
        "dims": {"Platforms": d.platforms, "Methods": d.methods, "Noise": d.noise, "Phenomena": d.phenomena,
                 "Metrics": d.metrics, "Qubit counts": [str(q) for q in d.qubit_counts]},
        "counts": {k: sum(1 for s in record.statements if s.kind == k) for k in ("direct", "interpretation", "unverified")},
    }


def build_corpus_view(result, titles: Dict[int, str]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    pid = lambda i: f"PUB-{i}"
    return {
        "papers": result.papers, "excluded": result.excluded, "not_analysed": result.not_analysed,
        "generated_at": result.generated_at[:16].replace("T", " "),
        "themes": [{"label": t.label, "papers": [{"id": pid(i), "title": titles.get(i, "")} for i in t.publication_ids]}
                   for t in result.themes],
        "untried": result.untried_in_corpus,
        "recurring": [{"field": r.field, "representative": r.representative, "papers": r.papers,
                       "members": [{"id": pid(m["publication_id"]), "title": m["title"], "text": m["text"], "kind": m["kind"],
                                    "page": m["page"]} for m in r.members]} for r in result.recurring],
        "method_platform": [{"row": c.row, "col": c.col, "n": len(c.publication_ids)} for c in result.method_platform],
        "method_noise": [{"row": c.row, "col": c.col, "n": len(c.publication_ids)} for c in result.method_noise],
    }
