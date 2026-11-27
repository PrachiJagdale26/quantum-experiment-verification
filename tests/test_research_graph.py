"""
Phase 7: the orchestrated research run (offline). Collectors and the model are fakes; every other step is real.
Restart is simulated by building a NEW runner (new graph, new SQLite connection) on the same checkpoint file.
"""

import json

import pytest

from src.agents.research_graph import ResearchRunner, Services
from src.experiment_planning import apply_edit, decide, load_plan
from src.gap_discovery import NoveltyChecker
from src.literature.ingestion import LiteratureIngestor
from src.research_analysis import PaperAnalyzer
from src.storage import Publication, ReportRecord, session_scope
from src.verification.engine import VerificationEngine
from tests.test_gap_discovery import FakeCollector, StableEmbed, bow_embed, rec

ABS = ("Here we prepare GHZ entanglement with error mitigation on superconducting qubits. "
       "Our study is limited to shallow circuits with few qubits because dephasing destroys entanglement.")


class NoLLM:
    demo_mode = True


@pytest.fixture
def setup(tmp_path):
    db = f"sqlite:///{tmp_path / 'r.db'}"
    for t in ("GHZ mitigation on superconducting qubits", "GHZ mitigation on trapped ions"):
        with session_scope(db) as s:
            p = Publication(title=t, normalized_title=t.lower(), source="openalex", abstract=ABS.replace(
                "superconducting qubits", "trapped ions") if "ions" in t else ABS, year=2023)
            s.add(p)
    return db, str(tmp_path / "ck.db")


def services(db, outage=False, per_query=10):
    noise = [rec("openalex", f"Unrelated optics paper {i}", f"A study of lasers {i}.", doi=f"10.1/q{i}") for i in range(20)]
    cols = {n: FakeCollector(n, noise, status="failed" if outage else "ok") for n in ("openalex", "arxiv", "crossref")}
    ing = LiteratureIngestor(db, collectors=cols)
    checker = NoveltyChecker(db, ingestor=ing, embed=StableEmbed(), citations=False, per_query=per_query)
    return Services(llm=NoLLM(), ingestor=ing, engine=VerificationEngine(), analyzer=PaperAnalyzer(None, db),
                    checker=checker, embed=bow_embed), cols


PARAMS = {"collect": False, "analyse_n": 5, "top_gaps": 5}


def test_run_pauses_for_researcher_and_resumes_after_restart(setup):
    db, ck = setup
    sv, _ = services(db)
    run = ResearchRunner(sv, db, ck).start(PARAMS)
    st = ResearchRunner(sv, db, ck).status(run)
    assert st["status"] == "waiting" and st["waiting"]["kind"] == "select_gap" and st["candidates"]
    assert [e["node"] for e in st["log"]] == ["collect", "analyse", "gaps"]
    assert "Analysed 2 papers" in st["log"][1]["summary"] and all("ms" in e for e in st["log"])

    gap = st["candidates"][0]["id"]
    sv2, _ = services(db)                                        # "restart": new services, graph and connection
    r2 = ResearchRunner(sv2, db, ck)
    r2.resume(run, f"CG-{gap}")
    st = r2.status(run)
    assert st["status"] == "waiting" and st["waiting"]["kind"] == "review_plan" and st["gap_id"] == gap
    assert st["novelty"]["classification"] in ("potentially_unexplored", "partially_investigated", "already_investigated")
    lin = st["lineage"]

    # Continuing without a decision waits again; the graph never decides for the researcher.
    r2.resume(run, "continue")
    assert r2.status(run)["waiting"]["kind"] == "review_plan" and load_plan(lin, db)["status"] == "draft"

    apply_edit(lin, {"research_question": "Q?", "hypothesis": "H.", "methodology": ["m"], "dependent_variables": ["P"],
                     "metrics": ["TVD"], "circuit": {"num_qubits": 3, "gates": [
                         {"name": "H", "target_qubits": [0]}, {"name": "CX", "target_qubits": [0, 1]},
                         {"name": "CX", "target_qubits": [0, 2]}],
                         "expected_outcomes": [{"basis_state": "000", "expected_probability": 0.5},
                                               {"basis_state": "111", "expected_probability": 0.5}]}}, db_url=db)
    decide(lin, "approved", db_url=db)
    ResearchRunner(sv2, db, ck).resume(run, "continue")
    st = ResearchRunner(sv2, db, ck).status(run)
    assert st["status"] == "done" and st["verdict"] == "pass" and st["outcome"] == "Plan verified in simulation"
    assert [e["node"] for e in st["log"]][-3:] == ["review", "verify", "report"]
    with session_scope(db) as s:
        assert s.get(ReportRecord, st["report_id"]).payload["verdict"] == "pass"


def test_low_coverage_requeries_at_most_twice_with_wider_search(setup):
    db, ck = setup
    sv, cols = services(db, outage=True)
    r = ResearchRunner(sv, db, ck)
    run = r.start({**PARAMS, "top_gaps": 1})
    r.resume(run, r.status(run)["candidates"][0]["id"])
    st = r.status(run)
    novelty = [e for e in st["log"] if e["node"] == "novelty"]
    assert len(novelty) == 3                                       # first try + 2 re-queries, then it moves on
    assert any("30 results per query" in c for c in novelty[-1]["calls"])
    assert st["novelty"]["classification"] == "insufficient_coverage" and st["waiting"]["kind"] == "review_plan"


def test_failure_is_recorded_and_retry_reruns_the_node(setup):
    db, ck = setup
    sv, _ = services(db)
    r = ResearchRunner(sv, db, ck)
    run = r.start(PARAMS)
    original = sv.checker.check
    sv.checker.check = lambda gid, **kw: (_ for _ in ()).throw(ConnectionError("network down"))
    r.resume(run, r.status(run)["candidates"][0]["id"])
    st = r.status(run)
    assert st["status"] == "failed" and st["current"] == "novelty" and "network down" in st["error"]
    assert st["failures"][0]["node"] == "novelty"
    sv.checker.check = original
    r.retry(run)
    assert r.status(run)["status"] == "waiting" and r.status(run)["waiting"]["kind"] == "review_plan"


def test_rejected_plan_ends_the_run_and_failed_verification_without_model_is_not_revised(setup):
    db, ck = setup
    sv, _ = services(db)
    r = ResearchRunner(sv, db, ck)
    run = r.start(PARAMS)
    r.resume(run, r.status(run)["candidates"][0]["id"])
    lin = r.status(run)["lineage"]
    decide(lin, "rejected", note="Not worth running now.", db_url=db)
    r.resume(run, "continue")
    st = r.status(run)
    assert st["status"] == "done" and st["outcome"] == "Plan rejected by you" and st["verdict"] is None

    run2 = r.start({**PARAMS, "gap_id": st["gap_id"], "analyse_n": 0})    # gap chosen up front: no pause
    st2 = r.status(run2)
    assert st2["waiting"]["kind"] == "review_plan"
    lin2 = st2["lineage"]
    apply_edit(lin2, {"research_question": "Q?", "hypothesis": "H.", "methodology": ["m"], "dependent_variables": ["P"],
                      "metrics": ["TVD"], "circuit": {"num_qubits": 1, "gates": [{"name": "H", "target_qubits": [0]}],
                                                      "expected_outcomes": [{"basis_state": "0", "expected_probability": 1.0}]}},
               db_url=db)
    decide(lin2, "approved", db_url=db)
    r.resume(run2, "continue")
    st2 = r.status(run2)
    assert st2["verdict"] == "fail" and st2["status"] == "done" and st2["outcome"] == "Plan failed verification"
    assert next(e for e in st2["log"] if e["node"] == "revise")["summary"].startswith("No live model")
