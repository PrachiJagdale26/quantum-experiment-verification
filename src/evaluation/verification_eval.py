"""
Quantum verification evaluation: false-pass and false-fail rates on cases with known answers.

Two sets:
  1. the 6 existing proposal benchmark cases (data/benchmark_cases, expected status written by hand);
  2. plan-verification cases whose ground truth comes from textbook theory, written here and never taken from the
     simulator: Bell and GHZ states, a GHZ prediction with the wrong populations, a GHZ-minus target (same
     probabilities, different phase), RY rotations (P(1) = sin^2(theta/2)), a wrong rotation prediction, impossible
     noise (T2 > 2 T1), noisy predictions that should and should not survive the stated noise, and a plan without a
     circuit (must be held, not passed or failed).
False pass = a case that should fail or be held was passed. False fail = a case that should pass was failed.
"""

import math
from typing import Any, Dict, List, Optional

from ..experiment_planning.schema import CircuitDraft, Parameter, ResearchProposal
from ..schemas.proposal import ExpectedOutcome, GateOperation, ProposalSpecification
from ..verification.engine import VerificationEngine
from ..verification.plan_verifier import extra_checks, plan_to_specification
from ..verification.verdict import overall_verdict
from .benchmark import load_benchmark_cases

S = 2 ** -0.5
G = lambda n, q, *p: GateOperation(name=n, target_qubits=list(q), params=list(p) or None)
O = lambda **kv: [ExpectedOutcome(basis_state=k[1:], expected_probability=v) for k, v in kv.items()]
GHZ = [G("H", [0]), G("CX", [0, 1]), G("CX", [0, 2])]
BELL = [G("H", [0]), G("CX", [0, 1])]
th = math.pi / 3


def plan_cases() -> List[Dict[str, Any]]:
    c = lambda n, gates, out, **kw: CircuitDraft(num_qubits=n, gates=gates, expected_outcomes=out, **kw)
    dep = lambda v, src="researcher": Parameter(name="Two-qubit depolarizing probability", value=v, source=src)
    return [
        {"id": "bell", "expect": "pass", "circuit": c(2, BELL, O(b00=0.5, b11=0.5), expected_state={"00": [S, 0], "11": [S, 0]})},
        {"id": "ghz", "expect": "pass", "circuit": c(3, GHZ, O(b000=0.5, b111=0.5), expected_state={"000": [S, 0], "111": [S, 0]})},
        {"id": "ghz wrong populations", "expect": "fail", "circuit": c(3, GHZ, O(b000=1.0))},
        {"id": "ghz-minus target (phase)", "expect": "fail", "circuit": c(3, GHZ, O(b000=0.5, b111=0.5),
                                                                       expected_state={"000": [S, 0], "111": [-S, 0]})},
        {"id": "ry(pi/3)", "expect": "pass", "circuit": c(1, [G("RY", [0], th)],
                                                         O(b0=round(math.cos(th / 2) ** 2, 4), b1=round(math.sin(th / 2) ** 2, 4)))},
        {"id": "ry(pi/3) wrong prediction", "expect": "fail", "circuit": c(1, [G("RY", [0], th)], O(b0=0.5, b1=0.5))},
        {"id": "x then h", "expect": "pass", "circuit": c(1, [G("X", [0]), G("H", [0])], O(b0=0.5, b1=0.5),
                                                          expected_state={"0": [S, 0], "1": [-S, 0]})},
        {"id": "impossible noise T2 > 2T1", "expect": "fail", "circuit": c(2, BELL, O(b00=0.5, b11=0.5)),
         "params": [Parameter(name="T1", value="50", unit="us", source="researcher"),
                    Parameter(name="T2", value="150", unit="us", source="researcher")]},
        {"id": "noisy prediction survives weak noise", "expect": "pass",
         "circuit": c(2, BELL, O(b00=0.5, b11=0.5), expected_outcomes_noisy=O(b00=0.5, b11=0.5)), "params": [dep("0.001")]},
        {"id": "noisy prediction fails 10% readout", "expect": "fail",
         "circuit": c(2, BELL, O(b00=0.5, b11=0.5), expected_outcomes_noisy=O(b00=0.5, b11=0.5)),
         "params": [Parameter(name="Readout error", value="0.1", source="researcher")]},
        {"id": "no circuit", "expect": "hold", "circuit": None},
        {"id": "circuit without predictions", "expect": "hold", "circuit": c(2, BELL, [])},
    ]


def _verdict(engine: VerificationEngine, case: Dict[str, Any]) -> str:
    p = ResearchProposal(gap_id=1, gap_statement="evaluation case", title=case["id"], parameters=case.get("params", []),
                         circuit=case["circuit"])
    report = engine.verify_proposal(plan_to_specification(p))
    extra, _, _ = extra_checks(p, report)
    report.checks.extend(extra)
    return overall_verdict(report)


def _rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    should_not_pass = [r for r in rows if r["expected"] != "pass"]
    should_pass = [r for r in rows if r["expected"] == "pass"]
    return {"cases": len(rows), "correct": sum(r["expected"] == r["got"] for r in rows),
            "false_pass": sum(r["got"] == "pass" for r in should_not_pass), "of_should_not_pass": len(should_not_pass),
            "false_fail": sum(r["got"] == "fail" for r in should_pass), "of_should_pass": len(should_pass), "cases_detail": rows}


def evaluate_verification(engine: Optional[VerificationEngine] = None) -> Dict[str, Any]:
    engine = engine or VerificationEngine()
    mapping = {"PASS": "pass", "FAIL": "fail", "INCONCLUSIVE": "hold"}
    bench = []
    for case in load_benchmark_cases():
        try:
            got = overall_verdict(engine.verify_proposal(ProposalSpecification(**case["proposal"])))
        except Exception:                                  # a malformed proposal that cannot even be parsed is a fail
            got = "fail"
        bench.append({"id": case["id"], "expected": mapping[case["expected_overall_status"]], "got": got})
    plans = [{"id": c["id"], "expected": c["expect"], "got": _verdict(engine, c)} for c in plan_cases()]
    return {"procedure": __doc__.strip(), "proposal_benchmark": _rates(bench), "plan_cases": _rates(plans)}
