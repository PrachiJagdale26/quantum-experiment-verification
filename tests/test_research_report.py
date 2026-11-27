"""
Phase 8: the 17-section research report with provenance labels, in Markdown, JSON and PDF (offline).
"""

import json

import pytest

from src.experiment_planning import ProposalPlanner, apply_edit, decide, save_draft
from src.gap_discovery import GapGenerator, NoveltyChecker, save_gaps
from src.literature.ingestion import LiteratureIngestor
from src.reporting.research_report import TITLES, build_report, save_report, to_markdown, to_pdf
from src.research_analysis import PaperAnalyzer
from src.storage import Publication, ReportRecord, session_scope
from src.verification.engine import VerificationEngine
from src.verification.plan_verifier import verify_plan
from tests.test_experiment_planning import PlanLLM
from tests.test_gap_discovery import FakeCollector, StableEmbed, bow_embed, rec

LIM = "Our study is limited to shallow circuits with few qubits because dephasing destroys entanglement."


class AnalysisLLM:
    demo_mode = False
    last_call = {"provider": "fake", "model": "fake-1"}

    def complete(self, system, user, **kw):
        return json.dumps({"items": []})


@pytest.fixture
def investigation(tmp_path):
    db = f"sqlite:///{tmp_path / 'rep.db'}"
    ids = []
    for t, ab, doi in (("Error mitigation on superconducting qubits", "Here we apply error mitigation to entanglement on "
                        "superconducting qubits with 5 qubits. " + LIM, "10.1000/em-sc"),
                       ("Error mitigation on trapped ions", "Here we apply error mitigation to entanglement on trapped ions. " + LIM,
                        "10.1000/em-ti")):
        with session_scope(db) as s:
            p = Publication(title=t, normalized_title=t.lower(), source="openalex", abstract=ab, year=2023, doi=doi,
                            authors=["A. Author", "B. Author"])
            s.add(p)
            s.flush()
            ids.append(p.id)
    for pid in ids:
        PaperAnalyzer(AnalysisLLM(), db).analyse(pid)
    gap = next(g for g in save_gaps(GapGenerator(db, embed=bow_embed).generate(), db) if g.origin == "stated_limitation")
    prior = rec("openalex", "Dephasing-limited entanglement under mitigation",
                "We study entanglement limited by dephasing and decoherence with error mitigation.", doi="10.1000/prior")
    noise = [rec("openalex", f"Unrelated optics paper {i}", f"Lasers {i}.", doi=f"10.1/n{i}") for i in range(20)]
    cols = {"openalex": FakeCollector("openalex", [prior] + noise), "arxiv": FakeCollector("arxiv"),
            "crossref": FakeCollector("crossref", noise)}
    NoveltyChecker(db, ingestor=LiteratureIngestor(db, collectors=cols), embed=StableEmbed(), citations=False).check(gap.id)
    lin = save_draft(ProposalPlanner(PlanLLM(ids[0]), db).draft(gap.id), db)
    apply_edit(lin, {"hypothesis": "Mitigation keeps GHZ populations at one half each.",
                     "circuit": {"num_qubits": 3, "gates": [{"name": "H", "target_qubits": [0]},
                                                            {"name": "CX", "target_qubits": [0, 1]},
                                                            {"name": "CX", "target_qubits": [0, 2]}],
                                 "expected_outcomes": [{"basis_state": "000", "expected_probability": 0.5},
                                                       {"basis_state": "111", "expected_probability": 0.5}]}},
               note="Added the circuit", db_url=db)
    decide(lin, "approved", note="Ready", db_url=db)
    verify_plan(lin, VerificationEngine(), db)
    return db, lin


def test_all_17_sections_with_provenance_and_references(investigation):
    db, lin = investigation
    r = build_report(lin, db, embed=bow_embed)
    assert [s["title"] for s in r["sections"]] == TITLES and len(TITLES) == 17
    assert all(s["items"] or any(t["rows"] for t in s["tables"]) for s in r["sections"])
    kinds = {it["kind"] for s in r["sections"] for it in s["items"]}
    assert {"quote", "generated", "computed", "researcher", "inference", "metadata"} <= kinds
    s11 = {it["text"].split(":")[0]: it for it in r["sections"][10]["items"]}
    assert s11["Hypothesis (to be tested, not a finding)"]["kind"] == "researcher"     # edited by the researcher
    assert s11["Research question"]["kind"] == "generated" and s11["Research question"]["source"] == "fake/planner-1"
    assert "pass" in r["sections"][12]["items"][0]["text"]
    dois = {x["doi"] for x in r["references"]}
    assert {"10.1000/em-sc", "10.1000/prior"} <= dois
    assert all(x["doi_url"] == f"https://doi.org/{x['doi']}" for x in r["references"] if x["doi"])
    # an unverified justification quote is labelled as generated, never as a quote
    just = [it for it in r["sections"][11]["items"] if it["text"].startswith("Justification") and "Invented" in it["text"]]
    assert just and just[0]["kind"] == "generated"


def test_markdown_pdf_json_and_record(investigation, tmp_path):
    db, lin = investigation
    r = build_report(lin, db, embed=bow_embed)
    md = to_markdown(r)
    for i, t in enumerate(TITLES, 1):
        assert f"## {i}. {t}" in md
    assert "*[Verified quote]*" in md and "*[Model-generated · fake/planner-1]*" in md
    assert "(https://doi.org/10.1000/prior)" in md
    pdf = to_pdf(r)
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000
    out = save_report(r, db, out_dir=str(tmp_path / "reports"))
    assert set(out["paths"]) == {"md", "json", "pdf"} and json.loads(out["json"])["plan_id"] == f"RP-{lin}"
    with session_scope(db) as s:
        assert s.get(ReportRecord, out["record_id"]).payload["sections"] == 17


def test_unverified_plan_says_so(investigation):
    db, lin = investigation
    apply_edit(lin, {"hypothesis": "Changed after verification."}, db_url=db)
    r = build_report(lin, db, embed=bow_embed)
    assert "The plan has changed since this run." in r["sections"][12]["items"][0]["text"]
    assert r["status"] == "draft"
