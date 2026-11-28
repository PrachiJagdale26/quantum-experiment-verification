"""End-to-end: papers -> gaps -> plan -> verification, including the single revision loop."""

import os

import pytest

from src.research.library import PaperIngestError
from src.research.planner import build_proposal
from src.research.domain import template_circuit
from src.schemas.proposal import ExpectedOutcome
from tests.research_fixtures import build_pipeline


@pytest.fixture(scope="module")
def pipe():
    p = build_pipeline()
    p.find_gaps()
    return p


def test_library_loads_and_analyses_sample_papers(pipe):
    papers = [p for _, p in pipe.library.papers()]
    assert len(papers) == 4
    assert all(p.limitations for p in papers)
    assert any(p.future_work for p in papers) and any(p.open_questions for p in papers)


def test_top_gap_plans_and_verifies(pipe):
    plan = pipe.plan(pipe.gaps[0].gap_id)
    assert plan.verification_status == "pass"
    report = pipe.reports[plan.plan_id]
    assert report.simulation is not None


def test_failed_plan_triggers_exactly_one_revision(pipe):
    gap = next(g for g in pipe.gaps if g.hints.get("family") == "ghz")
    plan = pipe.plan(gap.gap_id, verify=False)
    # Sabotage the prediction: claim 80/20 for a GHZ circuit.
    n = plan.proposal.circuit.num_qubits
    plan.proposal = build_proposal(plan.title, "obj", plan.hypothesis, plan.method_steps, "Trapped Ion", n,
                                   plan.proposal.circuit.gates,
                                   [ExpectedOutcome(basis_state="0" * n, expected_probability=0.8),
                                    ExpectedOutcome(basis_state="1" * n, expected_probability=0.2)])
    pipe.add_plan(plan)
    revised, report = pipe.verify(plan.plan_id)
    history = pipe.history[plan.plan_id]
    assert len(history) == 2, "expected the original attempt plus exactly one revision"
    assert history[0][0].verification_status == "fail"
    assert revised.revision == 1 and revised.revision_note
    assert revised.verification_status == "pass"
    assert {o.expected_probability for o in revised.proposal.expected_outcomes} == {0.5}


def test_unsupported_file_is_rejected(pipe):
    with pytest.raises(PaperIngestError):
        pipe.library.add_document(b"hello", "notes.xlsx")


def test_agenda_markdown_lists_gaps_and_plans(pipe):
    md = pipe.agenda_markdown()
    assert "# Research agenda" in md
    assert pipe.gaps[0].title in md
    assert "## Experiment plans" in md
