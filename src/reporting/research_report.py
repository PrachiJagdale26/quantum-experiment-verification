"""
Research investigation report (Module 8): the 17 sections, built from stored records only.

Every item states its provenance, so generated and verified content never look alike:
  quote       verbatim text verified against the stored source
  generated   written by a model (or a template) and not verified as a quote
  computed    produced deterministically (search counts, rules, simulation)
  inference   a conclusion drawn across several papers
  researcher  written or decided by the researcher
  metadata    bibliographic metadata from the source APIs
Publications are cited as [n]; section 17 lists them with DOI / arXiv links. Nothing absent is filled in: a
section without data says so.
"""

import json
import os
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import select

from ..experiment_planning import FEASIBILITY_LABELS, PARAM_SOURCE_LABELS, load_plan
from ..experiment_planning.schema import SECTIONS as PLAN_SECTIONS
from ..gap_discovery import CLASSIFICATION_LABELS, ORIGIN_LABELS, load_assessment, load_gap, load_gaps, score_gap, weighted_total
from ..research_analysis.cross_paper_analysis import analyse_corpus
from ..storage import Publication, ReportRecord, SearchRun, session_scope
from ..verification.plan_verifier import load_runs
from ..verification.verdict import overall_verdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KINDS = OrderedDict([("quote", "Verified quote"), ("generated", "Model-generated"), ("computed", "Computed"),
                     ("inference", "Inference across papers"), ("researcher", "Researcher"), ("metadata", "Metadata")])
TITLES = ["Selected research domain", "Literature collection summary", "Publications analysed", "Research themes discovered",
          "Cross-paper comparison", "Candidate research gaps", "Novelty assessments", "Supporting and contradicting evidence",
          "Closest related publications", "Selected research gap", "Research question and hypothesis",
          "Proposed experiment plan", "Quantum experiment verification", "Simulation results", "Limitations",
          "Researcher approval status", "References and DOI links"]
SYSTEM_LIMITATIONS = [
    "Novelty is assessed in the searched literature only (OpenAlex, arXiv, Crossref and this corpus); it never means unpublished.",
    "Paper analysis and comparisons mostly use abstracts; details that appear only in full texts can be missed.",
    "Research dimensions come from a curated ontology without word-sense disambiguation.",
    "Verification is ideal or modelled-noise simulation; it is not validation on quantum hardware.",
]


class _Refs:
    """Numbers publications in order of first citation."""
    def __init__(self, db_url):
        self.db_url, self.order = db_url, OrderedDict()

    def __call__(self, pid: Optional[int]) -> str:
        if not pid:
            return ""
        if pid not in self.order:
            self.order[pid] = len(self.order) + 1
        return f"[{self.order[pid]}]"

    def entries(self) -> List[Dict[str, Any]]:
        with session_scope(self.db_url) as s:
            pubs = {p.id: p for p in s.scalars(select(Publication).where(Publication.id.in_(list(self.order))))}
            out = []
            for pid, n in self.order.items():
                p = pubs.get(pid)
                if p is None:
                    continue
                authors = list(p.authors or [])
                out.append({"n": n, "publication_id": pid, "title": p.title, "year": p.year, "venue": p.venue,
                            "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                            "doi": p.doi, "doi_url": f"https://doi.org/{p.doi}" if p.doi else None,
                            "arxiv_url": f"https://arxiv.org/abs/{p.arxiv_id}" if p.arxiv_id else None,
                            "url": p.source_url, "kind": "metadata"})
            return out


def _item(text: str, kind: str, source: Optional[str] = None, refs: str = "") -> Dict[str, Any]:
    return {"text": text, "kind": kind, "source": source, "refs": refs}


def _table(columns: List[str], rows: List[List[Any]], kind: str = "computed") -> Dict[str, Any]:
    return {"columns": columns, "rows": [["" if c is None else str(c) for c in r] for r in rows], "kind": kind}


def _researcher_fields(history: List[Dict[str, Any]]) -> set:
    """Plan fields last written by the researcher (edits after the generated draft)."""
    edited = set()
    for h in history[1:]:
        if h.get("by") == "researcher":                 # model revisions record the model as the editor
            edited |= set(h.get("changed") or [])
    return edited


def build_report(lineage: int, db_url: Optional[str] = None, embed: Optional[Callable] = None) -> Dict[str, Any]:
    plan = load_plan(lineage, db_url)
    if plan is None:
        raise ValueError(f"No plan RP-{lineage}")
    p = plan["proposal"]
    gap = load_gap(p.gap_id, db_url)
    a = load_assessment(p.gap_id, db_url) if gap else None
    runs = load_runs(lineage, db_url)
    run = runs[0] if runs else None
    cite = _Refs(db_url)
    edited = _researcher_fields(plan["history"])
    gen_src = p.generated_by.replace(" + researcher edits", "")
    gen_kind = lambda field: "researcher" if field in edited else "generated"
    S: List[Dict[str, Any]] = [{"n": i + 1, "title": t, "items": [], "tables": []} for i, t in enumerate(TITLES)]

    with session_scope(db_url) as s:
        pubs = list(s.scalars(select(Publication).where(Publication.discovered_via == "collection",
                                                        Publication.is_synthetic.is_(False))))
        corpus = [{"id": x.id, "title": x.title, "year": x.year, "abstract": bool(x.abstract), "full": bool(x.full_text_available),
                   "excluded": x.curation_status == "excluded", "analysed": x.analysed_at is not None,
                   "retrieved": x.retrieved_at} for x in pubs]
        runs_c = list(s.scalars(select(SearchRun).where(SearchRun.purpose == "collection")))
        coll = [{"source": r.source, "status": r.status, "n": r.result_count, "query": r.query} for r in runs_c]

    # 1 domain
    queries = [q for q, _ in Counter(c["query"] for c in coll).most_common(8)]
    S[0]["items"] += [_item(f"Domain: {(gap and 'quantum physics') or 'quantum physics'}.", "metadata"),
                      _item("Searches that built the corpus: " + ("; ".join(f"“{q}”" for q in queries) if queries
                                                                      else "none recorded (papers were added individually)."), "computed")]
    # 2 collection
    by_src: Dict[str, List[int]] = {}
    for c in coll:
        v = by_src.setdefault(c["source"], [0, 0, 0])
        v[0] += 1
        v[1] += c["status"] in ("ok", "cached")
        v[2] += c["n"]
    S[1]["tables"].append(_table(["Source", "Searches", "Answered", "Records returned"], [[k, *v] for k, v in sorted(by_src.items())]))
    dates = sorted(c["retrieved"] for c in corpus if c["retrieved"])
    S[1]["items"] += [_item(f"{len(corpus)} publications in the corpus: {sum(c['abstract'] for c in corpus)} with abstracts, "
                            f"{sum(not c['abstract'] for c in corpus)} metadata only, {sum(c['full'] for c in corpus)} with full text, "
                            f"{sum(c['excluded'] for c in corpus)} excluded by the researcher.", "computed"),
                      _item(f"Retrieved between {dates[0]:%Y-%m-%d} and {dates[-1]:%Y-%m-%d}." if dates else "No retrieval dates.", "computed")]
    # 3 analysed
    an = [c for c in corpus if c["analysed"] and not c["excluded"]]
    S[2]["items"].append(_item(f"{len(an)} publications analysed ({sum(c['full'] for c in an)} from full text, the rest from abstracts). "
                               "Each statement was checked against the source text; unverified statements are never used as evidence.",
                               "computed"))
    S[2]["tables"].append(_table(["Ref", "Title", "Year", "Analysed from"],
                                 [[cite(c["id"]), c["title"], c["year"], "full text" if c["full"] else "abstract"] for c in an[:60]]))
    # 4-5 corpus analysis
    ca = analyse_corpus(db_url, embed=embed) if an else None
    if ca and ca.themes:
        for t in ca.themes:
            S[3]["items"].append(_item(f"{t.label}: " + " ".join(cite(i) for i in t.publication_ids), "inference"))
    else:
        S[3]["items"].append(_item("No themes: fewer than two analysed publications.", "computed"))
    if ca:
        cells = sorted(ca.method_platform, key=lambda c: -len(c.publication_ids))[:10]
        S[4]["tables"].append(_table(["Method", "Platform", "Papers"], [[c.row, c.col, " ".join(cite(i) for i in c.publication_ids)]
                                                                        for c in cells]))
        for u in ca.untried_in_corpus[:8]:
            S[4]["items"].append(_item(f"Not studied together in this corpus: {u['row']} × {u['col']} "
                                       f"({u['row_papers']} and {u['col_papers']} papers respectively).", "inference"))
        for r in ca.recurring[:5]:
            S[4]["items"].append(_item(f"Raised by {r.papers} papers ({r.field.replace('_', ' ')}): “{r.representative}” "
                                       + " ".join(cite(m["publication_id"]) for m in r.members), "quote"))
        if not ca.untried_in_corpus and not ca.recurring:
            S[4]["items"].append(_item("No untried combinations or recurring statements met the thresholds.", "computed"))
    # 6 candidate gaps
    gaps = load_gaps(db_url)
    rows = []
    for g in gaps:
        ga = load_assessment(g.id, db_url)
        rows.append([f"CG-{g.id}", g.title, ORIGIN_LABELS.get(g.origin, g.origin),
                     CLASSIFICATION_LABELS.get(ga["classification"], "") if ga else "Not checked",
                     f"{weighted_total(score_gap(g, ga)):.2f}", g.review_status])
    rows.sort(key=lambda r: -float(r[4]))
    S[5]["items"].append(_item(f"{len(gaps)} candidate gaps from verified statements and untried combinations; none is validated "
                               "unless the researcher marked it so.", "computed"))
    S[5]["tables"].append(_table(["Gap", "Title", "Origin", "Novelty", "Score", "Review"], rows[:25]))
    # 7 novelty
    assessed = [r for r in rows if r[3] != "Not checked"]
    S[6]["tables"].append(_table(["Gap", "Novelty classification", "Score"], [[r[0], r[3], r[4]] for r in assessed]))
    if a:
        cov, f = a["coverage"], a["findings"]
        S[6]["items"] += [
            _item(f"Selected gap CG-{gap.id}: {a['label']} (confidence {a['confidence']:.2f}). {f['reason']}", "computed"),
            _item(f"Confidence: {a['confidence_basis']['formula']} (coverage {a['confidence_basis']['coverage']}, evidence "
                  f"{a['confidence_basis']['evidence']}, agreement {a['confidence_basis']['agreement']}).", "computed"),
            _item(f"Searched {', '.join(cov['sources_ok']) or 'no source'}"
                  + (f" (failed: {', '.join(cov['sources_failed'])})" if cov["sources_failed"] else "")
                  + f"; {len(cov['queries'])} queries; {cov['screened']} publications screened; searched {(a['searched_at'] or '')[:10]}.",
                  "computed"),
            _item("Queries: " + "; ".join(f"{q['kind']}: “{q['query']}”" for q in cov["queries"]), "computed")]
    else:
        S[6]["items"].append(_item("The selected gap has not been checked for novelty.", "computed"))
    # 8 evidence
    for e in (gap.evidence if gap else []):
        kind = {"direct": "quote", "interpretation": "quote", "inference": "inference"}[e.kind]
        S[7]["items"].append(_item(f"Supporting: “{e.quote}”" if kind == "quote" else f"Supporting: {e.quote}", kind,
                                   refs=cite(e.publication_id)))
    if a:
        dims = list(a["findings"]["core_dimensions"])
        for c in a["findings"]["closest"]:
            if c["same"] >= max(1, len(dims) - 1):
                q = next((d["quote"] for d in c["dimensions"].values() if d["result"] == "same" and d.get("quote")), None)
                S[7]["items"].append(_item(f"Contradicting novelty: same on {c['same']} of {len(dims)} dimensions"
                                           + (f": “{q}”" if q else "."), "quote" if q else "computed", refs=cite(c["publication_id"])))
    if not S[7]["items"]:
        S[7]["items"].append(_item("No evidence recorded.", "computed"))
    # 9 closest
    if a:
        dims = list(a["findings"]["core_dimensions"])
        S[8]["tables"].append(_table(["Ref", "Title", "Year", "Similarity"] + [d.replace("_", " ") for d in dims],
                                     [[cite(c["publication_id"]), c["title"], c["year"], f"{c['similarity']:.2f}"]
                                      + [c["dimensions"].get(d, {}).get("result", "").replace("_", " ") for d in dims]
                                      for c in a["findings"]["closest"]]))
    else:
        S[8]["items"].append(_item("Not available: no novelty investigation.", "computed"))
    # 10 selected gap
    if gap:
        S[9]["items"] += [
            _item(f"CG-{gap.id} ({ORIGIN_LABELS.get(gap.origin, gap.origin)}): {gap.statement}",
                  "inference" if gap.origin == "unexplored_combination" else "quote",
                  refs=" ".join(cite(e.publication_id) for e in gap.evidence if e.role == "supporting")),
            _item("Dimensions: " + "; ".join(f"{d}: {', '.join(v.values)} ({v.source.replace('_', ' ')})"
                                             for d, v in gap.dimensions.items()), "computed"),
            _item(f"Researcher review of the gap: {gap.review_status}.", "researcher")]
        if gap.research_question:
            S[9]["items"].append(_item(f"Rewritten as a question: {gap.research_question}", "generated", gap.research_question_source))
    # 11 RQ & hypothesis
    for fld, label in (("research_question", "Research question"), ("hypothesis", "Hypothesis (to be tested, not a finding)")):
        val = getattr(p, fld)
        S[10]["items"].append(_item(f"{label}: {val}" if val else f"{label}: not written.", gen_kind(fld), gen_src))
    # 12 plan
    S[11]["items"].append(_item(f"{p.title} (RP-{lineage}, revision {plan['revision']}). Feasibility: "
                                f"{FEASIBILITY_LABELS[p.feasibility_class]}. {p.feasibility_reason}",
                                "researcher" if "feasibility_class" in edited else "computed"))
    for key, label, kind in PLAN_SECTIONS:
        if kind == "special" or key in ("gap_statement", "research_question", "hypothesis"):
            continue
        val = getattr(p, key)
        if val:
            S[11]["items"].append(_item(f"{label}: " + ("; ".join(val) if isinstance(val, list) else val), gen_kind(key), gen_src))
    S[11]["tables"].append(_table(["Parameter", "Value", "Source", "Basis"],
                                  [[x.name, f"{x.value} {x.unit}".strip(), PARAM_SOURCE_LABELS[x.source],
                                    (cite(x.publication_id) + " " if x.publication_id else "") + x.basis] for x in p.parameters]))
    for j in p.literature_justification:
        S[11]["items"].append(_item(f"Justification ({j.role.replace('_', ' ')}): {j.point}" + (f" “{j.quote}”" if j.quote else ""),
                                    "quote" if j.quote and j.verified else "generated" if j.quote else "inference",
                                    None if j.verified else gen_src, cite(j.publication_id)))
    if p.missing_information:
        S[11]["items"].append(_item("Still missing: " + "; ".join(p.missing_information), "computed"))
    # 13-14 verification
    if run:
        rep = run["report"]
        S[12]["items"] += [
            _item(f"Verdict on revision {run['revision']}: {overall_verdict(rep)} ({run['simulator']}, {(run['at'] or '')[:16]} UTC)."
                  + (" The plan has changed since this run." if run["revision"] != plan["revision"] else ""), "computed"),
            _item(rep.overall_summary, "computed")]
        S[12]["tables"].append(_table(["Check", "Name", "Result", "Details"],
                                      [[c.check_id, c.name, c.status.value, c.details[:300]] for c in rep.checks]))
        if rep.simulation:
            pred = {o.basis_state: o.expected_probability for o in run["spec"].expected_outcomes}
            states = sorted(set(rep.simulation.probabilities) | set(pred))
            S[13]["tables"].append(_table(["State", "Predicted", "Simulated (ideal)"],
                                          [[f"|{k}>", pred.get(k, ""), f"{rep.simulation.probabilities.get(k, 0):.4f}"] for k in states]))
        S[13]["items"].append(_item(f"Total variation distance: {rep.total_variation_distance}; state fidelity: "
                                    f"{f'{rep.fidelity_score:.4f}' if rep.fidelity_score is not None else 'not computed (no expected state)'}.",
                                    "computed"))
        if run["noise"]:
            S[13]["tables"].append(_table(["Noise condition", "Values (source)", "Top outcomes", "TVD to ideal"],
                                          [[r["label"], "; ".join(sorted(set(r["sources"].values()))),
                                            ", ".join(f"{k}: {v:.3f}" for k, v in sorted(r["probabilities"].items(), key=lambda x: -x[1])[:4]),
                                            r["tvd_to_ideal"]] for r in run["noise"]]))
    else:
        S[12]["items"].append(_item("Not verified yet: verification runs only on an approved revision.", "computed"))
        S[13]["items"].append(_item("No simulation results.", "computed"))
    # 15 limitations
    for l in p.limitations:
        S[14]["items"].append(_item(l, gen_kind("limitations"), gen_src))
    if a:
        S[14]["items"].append(_item(a["uncertainty"], "computed"))
    for l in SYSTEM_LIMITATIONS:
        S[14]["items"].append(_item(l, "computed"))
    # 16 approval
    S[15]["items"].append(_item(f"Plan RP-{lineage} is {plan['status']} (revision {plan['revision']}).", "researcher"))
    for d in plan["approvals"]:
        S[15]["items"].append(_item(f"{d['decision'].replace('_', ' ').capitalize()} on revision {d['revision']}, "
                                    f"{(d['at'] or '')[:16]}" + (f": “{d['note']}”" if d.get("note") else "."), "researcher"))
    S[15]["tables"].append(_table(["Revision", "By", "Changed", "Note"],
                                  [[h["revision"], h["by"], ", ".join(h["changed"]), h.get("note") or ""] for h in plan["history"]],
                                  kind="researcher"))
    if gap:
        S[15]["items"].append(_item(f"Gap CG-{gap.id} review status: {gap.review_status}.", "researcher"))
    # 17 references
    refs = cite.entries()
    S[16]["items"].append(_item(f"{len(refs)} publications cited; links point to the DOI, arXiv or source record.", "metadata"))
    return {"title": f"Research investigation: {p.title}", "plan_id": f"RP-{lineage}", "lineage": lineage,
            "gap_id": f"CG-{p.gap_id}", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": plan["status"], "revision": plan["revision"], "kinds": dict(KINDS), "sections": S, "references": refs}


# ------------------------------------------------------------------ renderers
def to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, default=str)


def _md_cell(v: str) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def to_markdown(report: Dict[str, Any]) -> str:
    out = [f"# {report['title']}", "",
           f"Plan {report['plan_id']} (revision {report['revision']}, {report['status']}) · gap {report['gap_id']} · "
           f"generated {report['generated_at']}", "",
           "**Provenance labels:** " + " · ".join(f"*[{v}]*" for v in report["kinds"].values())
           + ". Only *[Verified quote]* text is taken verbatim from a source; *[Model-generated]* text is a draft to check.", ""]
    for s in report["sections"]:
        out += [f"## {s['n']}. {s['title']}", ""]
        for it in s["items"]:
            tag = report["kinds"][it["kind"]] + (f" · {it['source']}" if it.get("source") and it["kind"] == "generated" else "")
            out.append(f"- *[{tag}]* {it['text']}" + (f" {it['refs']}" if it.get("refs") else ""))
        for t in s["tables"]:
            if not t["rows"]:
                continue
            out += ["", "| " + " | ".join(t["columns"]) + " |", "|" + "---|" * len(t["columns"])]
            out += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in t["rows"]]
        if s["n"] == 17:
            for r in report["references"]:
                link = r["doi_url"] or r["arxiv_url"] or r["url"] or ""
                out.append(f"{r['n']}. {r['authors'] + ', ' if r['authors'] else ''}{r['title']}"
                           f"{', ' + r['venue'] if r['venue'] else ''} ({r['year'] or 'n.d.'})."
                           + (f" [{('doi:' + r['doi']) if r['doi'] else link}]({link})" if link else ""))
        out.append("")
    return "\n".join(out)


PDF_COLORS = {"quote": "#1f7a4d", "generated": "#b45309", "computed": "#2457a6", "inference": "#6d3fb3",
              "researcher": "#0f766e", "metadata": "#555555"}


def to_pdf(report: Dict[str, Any]) -> bytes:
    from io import BytesIO
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font, bold = "Helvetica", "Helvetica-Bold"
    for reg, bd in ((r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
                    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if os.path.exists(reg) and os.path.exists(bd):
            try:
                pdfmetrics.registerFont(TTFont("RptSans", reg))
                pdfmetrics.registerFont(TTFont("RptSans-Bold", bd))
                font, bold = "RptSans", "RptSans-Bold"
                break
            except Exception:
                pass

    def safe(t: str) -> str:
        t = str(t).replace("⟩", ">").replace("⟨", "<").replace("ψ", "psi").replace("θ", "theta")
        if font == "Helvetica":
            t = t.encode("latin-1", "replace").decode("latin-1")
        return escape(t)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName=font, fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName=bold, fontSize=16, leading=20)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=12, leading=15, spaceBefore=10)
    story = [Paragraph(safe(report["title"]), h1),
             Paragraph(safe(f"Plan {report['plan_id']} (revision {report['revision']}, {report['status']}) · gap "
                            f"{report['gap_id']} · generated {report['generated_at']}"), small), Spacer(1, 4),
             Paragraph("Provenance: " + " · ".join(f'<font color="{PDF_COLORS[k]}"><b>[{safe(v)}]</b></font>'
                                                    for k, v in report["kinds"].items()), small), Spacer(1, 6)]
    for s in report["sections"]:
        story.append(Paragraph(safe(f"{s['n']}. {s['title']}"), h2))
        for it in s["items"]:
            tag = report["kinds"][it["kind"]] + (f" · {it['source']}" if it.get("source") and it["kind"] == "generated" else "")
            txt = safe(it["text"]) + (f" {safe(it['refs'])}" if it.get("refs") else "")
            if it["kind"] == "generated":
                txt = f"<i>{txt}</i>"
            story.append(Paragraph(f'<font color="{PDF_COLORS[it["kind"]]}"><b>[{safe(tag)}]</b></font> {txt}', body))
        for t in s["tables"]:
            if not t["rows"]:
                continue
            data = [[Paragraph(f"<b>{safe(c)}</b>", small) for c in t["columns"]]]
            data += [[Paragraph(safe(c), small) for c in r] for r in t["rows"]]
            tbl = Table(data, repeatRows=1, colWidths=[(180 * mm) / len(t["columns"])] * len(t["columns"]))
            tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                                     ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                                     ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [Spacer(1, 4), tbl]
        if s["n"] == 17:
            for r in report["references"]:
                link = r["doi_url"] or r["arxiv_url"] or r["url"]
                txt = safe(f"[{r['n']}] {r['authors'] + ', ' if r['authors'] else ''}{r['title']}"
                           f"{', ' + r['venue'] if r['venue'] else ''} ({r['year'] or 'n.d.'}).")
                if link:
                    txt += f' <link href="{escape(link)}" color="blue">{safe(("doi:" + r["doi"]) if r["doi"] else link)}</link>'
                story.append(Paragraph(txt, body))
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                      title=report["title"]).build(story)
    return buf.getvalue()


def save_report(report: Dict[str, Any], db_url: Optional[str] = None, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Write MD/JSON/PDF files and record them; returns {paths, record_id, md, json, pdf}."""
    out_dir = out_dir or os.path.join(ROOT, "data", "reports")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{report['plan_id']}_rev{report['revision']}_{report['generated_at'][:19].replace(':', '').replace('-', '')}"
    md, js, pdf = to_markdown(report), to_json(report), to_pdf(report)
    paths = {}
    for ext, data in (("md", md), ("json", js), ("pdf", pdf)):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        with open(path, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        paths[ext] = path
    with session_scope(db_url) as s:
        r = ReportRecord(kind="research_investigation", formats=["md", "json", "pdf"], path=paths["md"],
                         payload={"plan_id": report["plan_id"], "revision": report["revision"], "title": report["title"],
                                  "generated_at": report["generated_at"], "sections": len(report["sections"]),
                                  "references": len(report["references"]), "paths": paths})
        s.add(r)
        s.flush()
        rid = r.id
    return {"paths": paths, "record_id": rid, "md": md, "json": js, "pdf": pdf}
