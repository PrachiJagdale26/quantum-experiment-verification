"""
Phase 6: verifying approved plans: adapter, ideal checks, state fidelity, noise conditions, assumptions,
hardware status, and the rule that only an approved revision is verified (offline; no model).
"""

import numpy as np
import pytest

from src.experiment_planning import apply_edit, decide, save_draft
from src.experiment_planning.schema import CircuitDraft, Parameter, ResearchProposal
from src.quantum.noise import (NoiseCondition, conditions_from_parameters, noisy_simulation, parse_values, state_fidelity,
                               validate_condition)
from src.schemas.proposal import CircuitSpec, ExpectedOutcome, GateOperation
from src.storage import CandidateGap, session_scope
from src.verification.engine import VerificationEngine
from src.verification.plan_verifier import NotApproved, load_runs, plan_to_specification, verify_plan

GHZ = [GateOperation(name="H", target_qubits=[0]), GateOperation(name="CX", target_qubits=[0, 1]),
       GateOperation(name="CX", target_qubits=[0, 2])]
EVEN = [ExpectedOutcome(basis_state="000", expected_probability=0.5), ExpectedOutcome(basis_state="111", expected_probability=0.5)]
S = 2 ** -0.5


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'v.db'}"


@pytest.fixture
def engine():
    return VerificationEngine()


def _approved(db, circuit=None, params=(), feasibility="simulatable"):
    with session_scope(db) as s:
        g = CandidateGap(title="g", statement="s", origin="stated_limitation", generator="test", payload={"key": "k"})
        s.add(g)
        s.flush()
        gid = g.id
    p = ResearchProposal(gap_id=gid, gap_statement="s", title="GHZ under noise", research_question="Does it hold?",
                         hypothesis="It holds.", methodology=["Run"], dependent_variables=["P(000)"], metrics=["TVD"],
                         parameters=list(params), circuit=circuit, feasibility_class=feasibility)
    lin = save_draft(p, db)
    decide(lin, "approved", db_url=db)
    return lin


def _check(res, cid):
    return next(c for c in res["report"].checks if c.check_id == cid)


# ------------------------------------------------------------------- noise layer
def test_noise_physics():
    spec = CircuitSpec(num_qubits=3, gates=GHZ)
    ideal, _ = noisy_simulation(spec, NoiseCondition(label="none"))
    assert ideal == {"000": 0.5, "111": 0.5}
    flip, _ = noisy_simulation(CircuitSpec(num_qubits=2, gates=[GateOperation(name="X", target_qubits=[0])]),
                               NoiseCondition(label="r", readout=0.1))
    assert flip == pytest.approx({"00": 0.09, "01": 0.81, "10": 0.01, "11": 0.09})
    probs, rho = noisy_simulation(spec, NoiseCondition(label="d", depolarizing_2q=0.05))
    assert probs["000"] < 0.5 and abs(sum(probs.values()) - 1) < 1e-9
    assert 0.85 < state_fidelity({"000": [1, 0], "111": [1, 0]}, rho, 3) < 1
    relax, _ = noisy_simulation(spec, NoiseCondition(label="t", t1_us=20, t2_us=10))
    assert relax["000"] > relax["111"]                         # amplitude damping drives towards |0>
    assert validate_condition(NoiseCondition(label="x", t1_us=50, t2_us=150))
    assert parse_values("0.001-0.05") == [0.001, 0.05] and parse_values("1%") == [0.01]


def test_parameters_become_conditions_with_sources():
    conds, notes = conditions_from_parameters([
        Parameter(name="Two-qubit depolarizing probability", value="0.001-0.05", source="researcher"),
        Parameter(name="T1", value="100", unit="us", source="literature", verified=True),
        Parameter(name="Shots", value="8192", source="assumption")])
    assert [c.label for c in conds] == ["low end of every stated range", "high end of every stated range"]
    assert conds[0].depolarizing_2q == 0.001 and conds[1].depolarizing_2q == 0.05 and conds[0].depolarizing_1q == 0
    assert conds[0].t1_us == 100 and conds[0].gate_time_2q_ns == 300
    assert any("gate time" in n for n in notes)
    assert conditions_from_parameters([Parameter(name="Shots", value="8192", source="assumption")]) == ([], [])
    both, notes = conditions_from_parameters([Parameter(name="Error rate", value="0.01", source="researcher"),
                                              Parameter(name="Depolarizing probability", value="0.2", source="assumption")])
    assert both[0].depolarizing_1q == 0.01 and any("ignoring" in n and "0.2" in n for n in notes)


# ---------------------------------------------------------------- verification
def test_wrong_ghz_prediction_fails(db, engine):
    lin = _approved(db, CircuitDraft(num_qubits=3, gates=GHZ,
                                     expected_outcomes=[ExpectedOutcome(basis_state="000", expected_probability=1.0)]))
    res = verify_plan(lin, engine, db)
    assert res["verdict"] == "fail" and _check(res, "CHK-DIST-01").status.value == "FAIL"
    assert _check(res, "CHK-HW-01").status.value == "NOT_RUN" and "not behaviour on physical hardware" in _check(res, "CHK-HW-01").details


def test_correct_ghz_passes_with_fidelity(db, engine):
    lin = _approved(db, CircuitDraft(num_qubits=3, gates=GHZ, expected_outcomes=EVEN,
                                     expected_state={"000": [S, 0], "111": [S, 0]}))
    res = verify_plan(lin, engine, db)
    assert res["verdict"] == "pass"
    fid = _check(res, "CHK-FID-01")
    assert fid.status.value == "PASS" and fid.metadata["fidelity"] == pytest.approx(1.0)
    assert _check(res, "CHK-NOISE-01").status.value == "NOT_RUN"
    wrong = _approved(db, CircuitDraft(num_qubits=3, gates=GHZ, expected_outcomes=EVEN,
                                       expected_state={"000": [S, 0], "111": [-S, 0]}))      # GHZ-minus: probabilities agree
    res2 = verify_plan(wrong, engine, db)
    assert res2["verdict"] == "fail" and _check(res2, "CHK-FID-01").status.value == "FAIL"


def test_noisy_predictions_checked_and_assumptions_flagged(db, engine):
    lin = _approved(db, CircuitDraft(num_qubits=3, gates=GHZ, expected_outcomes=EVEN, expected_outcomes_noisy=EVEN),
                    params=[Parameter(name="Two-qubit depolarizing probability", value="0.001-0.05", source="researcher"),
                            Parameter(name="Readout error", value="0.1", source="assumption", basis="typical")])
    res = verify_plan(lin, engine, db)
    assert _check(res, "CHK-DIST-01").status.value == "PASS"            # ideal prediction is right
    assert _check(res, "CHK-NOISE-02").status.value == "FAIL"           # but it does not survive 10% readout error
    assert res["verdict"] == "fail"
    assert [r["label"] for r in res["noise"]] == ["low end of every stated range", "high end of every stated range"]
    assert res["noise"][1]["tvd_to_ideal"] > res["noise"][0]["tvd_to_ideal"]
    assert _check(res, "CHK-ASSUME-01").status.value == "INCONCLUSIVE" and "Readout error" in _check(res, "CHK-ASSUME-01").details


def test_noise_without_noisy_prediction_is_reported_not_judged(db, engine):
    lin = _approved(db, CircuitDraft(num_qubits=3, gates=GHZ, expected_outcomes=EVEN),
                    params=[Parameter(name="Depolarizing probability", value="0.01", source="researcher")])
    res = verify_plan(lin, engine, db)
    assert res["verdict"] == "pass" and _check(res, "CHK-NOISE-02").status.value == "NOT_RUN"
    assert "ideal prediction still holds" in _check(res, "CHK-NOISE-02").details


def test_impossible_noise_fails(db, engine):
    lin = _approved(db, CircuitDraft(num_qubits=3, gates=GHZ, expected_outcomes=EVEN),
                    params=[Parameter(name="T1", value="50", unit="us", source="researcher"),
                            Parameter(name="T2", value="150", unit="us", source="researcher")])
    res = verify_plan(lin, engine, db)
    assert res["verdict"] == "fail" and "physically impossible" in _check(res, "CHK-NOISE-01").details


def test_unspecified_plan_is_held_and_hardware_plans_say_so(db, engine):
    lin = _approved(db, None, feasibility="needs_hardware")
    spec = plan_to_specification(ResearchProposal(gap_id=1, gap_statement="s", title="t"))
    assert spec.circuit is None
    res = verify_plan(lin, engine, db)
    assert res["verdict"] == "hold"
    assert "hardware validation has not been performed" in _check(res, "CHK-HW-01").details


def test_only_approved_revisions_are_verified(db, engine):
    lin = _approved(db, CircuitDraft(num_qubits=3, gates=GHZ, expected_outcomes=EVEN))
    verify_plan(lin, engine, db)
    apply_edit(lin, {"hypothesis": "Changed."}, db_url=db)              # new draft revision
    with pytest.raises(NotApproved):
        verify_plan(lin, engine, db)
    runs = load_runs(lin, db)
    assert len(runs) == 1 and runs[0]["revision"] == 0 and runs[0]["verdict"] == "pass"
