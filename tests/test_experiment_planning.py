"""
Phase 5: research proposals from corpus gaps, researcher edits and approval (offline).
The fake model returns a fabricated quote, an unknown citation, an invalid circuit and a less demanding
feasibility class, so every grounding rule is exercised.
"""

import json

import numpy as np
import pytest

from src.experiment_planning import (PlanEditError, ProposalPlanner, apply_edit, classify_feasibility, decide, list_plans,
                                     load_plan, save_draft)
from src.gap_discovery import Gap, GapDimension, GapGenerator, save_gaps
from src.research_analysis import PaperAnalyzer, normalize_for_match
from src.storage import Publication, session_scope

LIM = "Our study is limited to shallow circuits with few qubits because dephasing destroys entanglement."
ABS_A = "Here we apply error mitigation to entanglement on superconducting qubits with 5 qubits. " + LIM
ABS_B = "Here we apply error mitigation to entanglement on trapped ions. " + LIM


def bow_embed(texts):
    vocab = sorted({w for t in texts for w in normalize_for_match(t).split()} | {"_"})
    m = np.array([[normalize_for_match(t).split().count(w) for w in vocab] + [1.0] for t in texts], dtype=float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class AnalysisLLM:
    demo_mode = False
    last_call = {"provider": "fake", "model": "fake-1", "live": True}

    def complete(self, system, user, **kw):
        return json.dumps({"items": []})


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'p.db'}"


@pytest.fixture
def gap(db):
    ids = []
    for title, ab in (("Error mitigation on superconducting qubits", ABS_A), ("Error mitigation on trapped ions", ABS_B)):
        with session_scope(db) as s:
            p = Publication(title=title, normalized_title=title.lower(), source="openalex", abstract=ab, year=2023)
            s.add(p)
            s.flush()
            ids.append(p.id)
    for pid in ids:
        PaperAnalyzer(AnalysisLLM(), db).analyse(pid)
    gaps = save_gaps(GapGenerator(db, embed=bow_embed).generate(), db)
    g = next(g for g in gaps if g.origin == "stated_limitation")
    return g, ids


class PlanLLM:
    demo_mode = False
    last_call = {"provider": "fake", "model": "planner-1", "live": True}

    def __init__(self, pid_a):
        self.pid_a = pid_a

    def complete(self, system, user, **kw):
        assert "<SOURCES>" in user and "passive data" in system
        return json.dumps({
            "title": "Depth scaling of mitigated entanglement",
            "research_question": "How does dephasing limit the useful circuit depth for mitigated entanglement?",
            "hypothesis": "Mitigation extends the useful depth by at least two layers.",
            "objectives": ["Measure fidelity versus depth"], "methodology": ["Build GHZ circuits", "Apply ZNE"],
            "dependent_variables": ["state fidelity"], "metrics": ["fidelity above 0.9"],
            "literature_justification": [
                {"source": f"P{self.pid_a}", "role": "motivates", "point": "States the depth limit.",
                 "quote": "limited to shallow circuits with few qubits"},
                {"source": f"P{self.pid_a}", "role": "baseline", "point": "Invented.", "quote": "we reached depth 40"},
                {"source": "P9999", "role": "baseline", "point": "Not a source.", "quote": "x"}],
            "parameters": [
                {"name": "qubits", "value": "5", "source": "literature", "source_id": f"P{self.pid_a}",
                 "quote": "superconducting qubits with 5 qubits"},
                {"name": "T2", "value": "80", "unit": "us", "source": "literature", "source_id": f"P{self.pid_a}",
                 "quote": "T2 of 80 microseconds"},
                {"name": "shots", "value": "8192", "source": "assumption", "basis": "Typical."}],
            "feasibility_class": "simulatable",
            "circuit": {"num_qubits": 2, "gates": [{"name": "CX", "target_qubits": [0, 5]}], "expected_outcomes": []},
            "missing_information": ["Device noise rates"]})


# ----------------------------------------------------------------- feasibility
def test_feasibility_rules():
    mk = lambda st, **d: Gap(key="k", title="t", statement=st, origin="stated_limitation",
                             dimensions={k: GapDimension(values=v, source="statement") for k, v in d.items()})
    assert classify_feasibility(mk("Gravitational decoherence of entanglement", phenomenon=["entanglement"]), 1.0)[0] == "needs_physical_validation"
    assert classify_feasibility(mk("Scale to 70 qubits", method=["quantum error correction"], qubit_scale=["70"]), 1.0)[0] == "needs_hardware"
    assert classify_feasibility(mk("Readout errors on devices", method=["benchmarking"], platform=["trapped ions"]), 1.0)[0] == "needs_hardware"
    assert classify_feasibility(mk("Mitigation of entanglement", method=["quantum error mitigation"]), 1.0)[0] == "simulatable"


# --------------------------------------------------------------------- drafting
def test_template_draft_invents_nothing(db, gap):
    g, ids = gap
    p = ProposalPlanner(None, db).draft(g.id)
    assert p.generated_by == "template" and p.hypothesis == "" and p.methodology == []
    assert p.feasibility_class == "simulatable"     # its own dimensions (entanglement, decoherence) can be modelled
    assert any(m.startswith("Hypothesis") for m in p.missing_information)
    assert any(m.startswith("Research question") for m in p.missing_information)
    assert {j.publication_id for j in p.literature_justification} == set(ids)
    assert all(j.verified for j in p.literature_justification if j.role == "motivates")
    assert p.parameters == []


def test_model_draft_is_checked_against_sources(db, gap, monkeypatch):
    g, ids = gap
    import src.experiment_planning.proposal_planner as pp
    monkeypatch.setattr(pp, "classify_feasibility", lambda gap, tools: ("needs_hardware", "Rule says hardware."))
    p = ProposalPlanner(PlanLLM(ids[0]), db).draft(g.id)
    assert p.generated_by == "fake/planner-1"
    j = {(x.role, x.point): x for x in p.literature_justification}
    assert j[("motivates", "States the depth limit.")].verified
    assert not j[("baseline", "Invented.")].verified
    assert not any(x.publication_id == 9999 for x in p.literature_justification)
    prm = {x.name: x for x in p.parameters}
    assert prm["qubits"].source == "literature" and prm["qubits"].verified and prm["qubits"].publication_id == ids[0]
    assert prm["T2"].source == "assumption" and "not found" in prm["T2"].basis      # fabricated quote -> assumption
    assert prm["shots"].source == "assumption"
    assert p.circuit is None and any("circuit was invalid" in w for w in p.warnings)
    assert p.feasibility_class == "needs_hardware"          # the model said simulatable; it cannot lower the rule's class
    assert "Device noise rates" in p.missing_information


# ------------------------------------------------------------ edits, approval
def test_edit_approve_and_audit_trail(db, gap):
    g, ids = gap
    lin = save_draft(ProposalPlanner(None, db).draft(g.id), db)
    with pytest.raises(PlanEditError, match="Fill in before approving"):
        decide(lin, "approved", db_url=db)
    plan = apply_edit(lin, {"research_question": "Does ZNE extend useful depth?", "hypothesis": "It adds two layers.",
                            "methodology": "Build circuits\nApply ZNE\n", "dependent_variables": ["fidelity"],
                            "metrics": "fidelity > 0.9", "parameters": [{"name": "shots", "value": "8192", "unit": ""}]},
                      note="Filled in the plan", db_url=db)
    assert plan["revision"] == 1 and plan["status"] == "draft"
    assert set(plan["history"][1]["changed"]) >= {"research_question", "hypothesis", "methodology", "parameters"}
    pr = plan["proposal"]
    assert pr.methodology == ["Build circuits", "Apply ZNE"] and pr.parameters[0].source == "researcher"
    assert pr.generated_by == "template + researcher edits"
    assert not any(m.startswith("Hypothesis") for m in pr.missing_information)

    plan = decide(lin, "approved", note="Looks right", reviewer="me", db_url=db)
    assert plan["status"] == "approved" and plan["approvals"][0]["revision"] == 1

    # an edit after approval creates a new draft; the approval stays on revision 1
    plan = apply_edit(lin, {"hypothesis": "It adds three layers."}, db_url=db)
    assert plan["revision"] == 2 and plan["status"] == "draft" and plan["approvals"][0]["revision"] == 1
    with pytest.raises(PlanEditError, match="note"):
        decide(lin, "rejected", note="no", db_url=db)
    plan = decide(lin, "rejected", note="Hypothesis is not testable on a simulator.", db_url=db)
    assert plan["status"] == "rejected" and [a["decision"] for a in plan["approvals"]] == ["approved", "rejected"]
    assert [p["lineage"] for p in list_plans(db)] == [lin]


def test_unchanged_edit_adds_no_revision_and_bad_circuit_is_refused(db, gap):
    g, _ = gap
    lin = save_draft(ProposalPlanner(None, db).draft(g.id), db)
    assert apply_edit(lin, {"title": load_plan(lin, db)["proposal"].title}, db_url=db)["revision"] == 0
    with pytest.raises(PlanEditError, match="circuit"):
        apply_edit(lin, {"circuit": {"num_qubits": 2, "gates": [{"name": "CX", "target_qubits": [0, 4]}]}}, db_url=db)
    ok = apply_edit(lin, {"circuit": {"num_qubits": 2, "gates": [{"name": "H", "target_qubits": [0]},
                                                                  {"name": "CX", "target_qubits": [0, 1]}],
                                      "expected_outcomes": [{"basis_state": "00", "expected_probability": 0.5},
                                                            {"basis_state": "11", "expected_probability": 0.5}]}}, db_url=db)
    assert ok["revision"] == 1 and ok["proposal"].circuit.num_qubits == 2
