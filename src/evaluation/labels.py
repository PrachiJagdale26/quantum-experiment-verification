"""
Labelling sheets for the metrics that need human judgement, and their scoring.

Sheets (CSV, one row per item; fill the label_* columns, leave a row blank to skip it):
  extraction_annotation.csv   model statements of up to 20 papers: is the field right? is it faithful to the source?
  gap_review.csv              candidate gaps: relevance and evidence support (1-5), duplicate of which gap
  plan_rubric.csv             research proposals: consistency, completeness, feasibility, alignment with the gap (1-5)
  future_work.csv             future-work / open-problem statements from papers 3+ years old: addressed later?
  partially_investigated.csv  published questions with one dimension changed: already / partially / unexplored?
A sheet that already exists is never overwritten (it may hold labels). A metric is reported only from labelled rows,
with the count; with none labelled it is "not measured".
"""

import csv
import os
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..experiment_planning import list_plans
from ..gap_discovery import load_assessment, load_gaps
from ..knowledge_base.ontology import CONCEPTS, find_concepts
from ..research_analysis.paper_analyzer import load_analysis
from ..storage import Publication, session_scope

SHEETS = {
    "extraction_annotation.csv": ["publication_id", "title", "doi", "field", "statement", "kind", "evidence",
                                  "label_field_correct (yes/no)", "label_faithful (yes/no)", "notes"],
    "gap_review.csv": ["gap", "title", "statement", "origin", "evidence", "novelty", "label_relevance (1-5)",
                       "label_evidence_support (1-5)", "label_duplicate_of", "notes"],
    "plan_rubric.csv": ["plan", "gap", "title", "research_question", "hypothesis", "methodology", "label_consistency (1-5)",
                        "label_completeness (1-5)", "label_feasibility (1-5)", "label_alignment (1-5)", "notes"],
    "future_work.csv": ["publication_id", "title", "year", "doi", "statement", "label_addressed_later (yes/no/unknown)",
                        "label_evidence_doi", "notes"],
    "partially_investigated.csv": ["source_publication_id", "source_title", "original_question", "changed_dimension",
                                   "from", "to", "modified_question", "label_status (already/partially/unexplored)", "notes"],
}


def _analysed(db_url) -> List[Publication]:
    with session_scope(db_url) as s:
        return list(s.scalars(select(Publication).where(Publication.analysed_at.is_not(None), Publication.is_synthetic.is_(False),
                                                        Publication.discovered_via == "collection",
                                                        Publication.curation_status != "excluded").order_by(Publication.id)))


def _rows(name: str, db_url: Optional[str]) -> List[List[Any]]:
    pubs = _analysed(db_url)
    if name == "extraction_annotation.csv":
        out = []
        for p in pubs[:20]:
            r = load_analysis(p.id, db_url)
            for st in r.statements if r else []:
                if st.extractor not in ("rule-based", "ontology"):
                    out.append([p.id, p.title, p.doi, st.field, st.text, st.kind, st.provenance.excerpt if st.provenance else "",
                                "", "", ""])
        return out
    if name == "gap_review.csv":
        return [[f"CG-{g.id}", g.title, g.statement, g.origin, " | ".join(e.quote for e in g.evidence)[:600],
                 (load_assessment(g.id, db_url) or {}).get("label", "not checked"), "", "", "", ""] for g in load_gaps(db_url)]
    if name == "plan_rubric.csv":
        return [[f"RP-{pl['lineage']}", f"CG-{pl['gap_id']}", pl["proposal"].title, pl["proposal"].research_question,
                 pl["proposal"].hypothesis, " | ".join(pl["proposal"].methodology), "", "", "", "", ""] for pl in list_plans(db_url)]
    if name == "future_work.csv":
        year = datetime.now(timezone.utc).year
        out = []
        for p in pubs:
            if p.year and p.year <= year - 3:
                r = load_analysis(p.id, db_url)
                out += [[p.id, p.title, p.year, p.doi, st.text, "", "", ""] for st in (r.statements if r else [])
                        if st.field in ("future_work", "unresolved_problem") and st.kind in ("direct", "interpretation")]
        return out
    if name == "partially_investigated.csv":
        out = []
        for p in pubs:
            r = load_analysis(p.id, db_url)
            q = next((st.text for st in (r.statements if r else []) if st.field in ("research_question", "objective", "research_problem")
                      and st.kind in ("direct", "interpretation")), None)
            hit = next((h for h in find_concepts(q or "") if h.category in ("platform", "method")), None)
            if not q or not hit:
                continue
            alt = next(c for c, (cat, _, _) in CONCEPTS.items() if cat == hit.category and c != hit.concept)
            out.append([p.id, p.title, q, hit.category, hit.concept, alt, q[:hit.start] + alt + q[hit.end:], "", ""])
            if len(out) >= 15:
                break
        return out
    return []


def create_sheets(db_url: Optional[str], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    status = {}
    for name, cols in SHEETS.items():
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            status[name] = "kept (may contain labels)"
            continue
        rows = _rows(name, db_url)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        status[name] = f"created with {len(rows)} rows"
    return status


def _read(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _col(rows, prefix):
    key = next((k for k in (rows[0].keys() if rows else []) if k.startswith(prefix)), None)
    return [r[key].strip().lower() for r in rows if key and r.get(key, "").strip()]


def _yes_share(vals: List[str]) -> Dict[str, Any]:
    yn = [v for v in vals if v in ("yes", "no")]
    return {"labelled": len(yn), "value": round(sum(v == "yes" for v in yn) / len(yn), 3) if yn else None}


def _mean(vals: List[str]) -> Dict[str, Any]:
    nums = [float(v) for v in vals if v.replace(".", "", 1).isdigit()]
    return {"labelled": len(nums), "value": round(mean(nums), 2) if nums else None}


def score_sheets(out_dir: str) -> Dict[str, Any]:
    ex = _read(os.path.join(out_dir, "extraction_annotation.csv"))
    gr = _read(os.path.join(out_dir, "gap_review.csv"))
    pr = _read(os.path.join(out_dir, "plan_rubric.csv"))
    fw = _read(os.path.join(out_dir, "future_work.csv"))
    pi = _read(os.path.join(out_dir, "partially_investigated.csv"))
    return {
        "extraction_field_accuracy": {**_yes_share(_col(ex, "label_field_correct")), "rows": len(ex)},
        "extraction_attribution_accuracy": {**_yes_share(_col(ex, "label_faithful")), "rows": len(ex)},
        "gap_relevance_mean": {**_mean(_col(gr, "label_relevance")), "rows": len(gr)},
        "gap_evidence_support_mean": {**_mean(_col(gr, "label_evidence_support")), "rows": len(gr)},
        "plan_rubric_means": {k: {**_mean(_col(pr, f"label_{k}")), "rows": len(pr)}
                              for k in ("consistency", "completeness", "feasibility", "alignment")},
        "future_work_addressed_share": {**_yes_share(_col(fw, "label_addressed_later")), "rows": len(fw)},
        "partially_investigated_labelled": {"labelled": len(_col(pi, "label_status")), "rows": len(pi),
                                            "value": None, "note": "Accuracy needs a live novelty run on the labelled rows."},
    }
