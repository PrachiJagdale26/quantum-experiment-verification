"""Phase 9: evaluation code (offline)."""

import csv

from src.evaluation.labels import SHEETS, score_sheets
from src.evaluation.verification_eval import evaluate_verification
from src.research_analysis.paper_analyzer import normalize_for_match, verify_excerpt

SRC = ("Here we prepare GHZ states on 12 superconducting qubits. The fidelity is 0.87 at depth 4, and crosstalk "
       "does not limit the protocol. Extending this to trapped ions remains open.")


def test_verifier_rejects_meaning_changes_but_accepts_genuine_variants():
    src = normalize_for_match(SRC)
    ok = lambda t: verify_excerpt(t, src, [])[0]
    assert ok("The fidelity is 0.87 at depth 4, and crosstalk does not limit the protocol.")
    assert ok("the fidelity is 0.87 at depth 4, and crosstalk does not limit the protocol")        # case
    assert ok("Here we prepare GHZ states on 12 superconducting  qubits.")                            # whitespace
    assert not ok("The fidelity is 0.97 at depth 4, and crosstalk does not limit the protocol.")       # number
    assert not ok("The fidelity is 0.87 at depth 4, and crosstalk does limit the protocol.")           # negation removed
    assert not ok("Here we prepare GHZ states on 12 trapped-ion qubits.")                              # term swapped
    assert not ok("Here we do not prepare GHZ states on 12 superconducting qubits.")                   # negation added


def test_verification_evaluation_has_no_false_passes_or_fails():
    r = evaluate_verification()
    for key in ("proposal_benchmark", "plan_cases"):
        assert r[key]["false_pass"] == 0 and r[key]["false_fail"] == 0 and r[key]["correct"] == r[key]["cases"]
    got = {c["id"]: c["got"] for c in r["plan_cases"]["cases_detail"]}
    assert got["ghz-minus target (phase)"] == "fail" and got["no circuit"] == "hold"


def test_label_scoring_counts_only_labelled_rows(tmp_path):
    def write(name, rows):
        with open(tmp_path / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(SHEETS[name])
            w.writerows(rows)
    blank = lambda n: [""] * n
    write("extraction_annotation.csv", [[1, "t", "", "result", "s", "direct", "e", "yes", "yes", ""],
                                        [1, "t", "", "result", "s", "direct", "e", "no", "yes", ""],
                                        [1, "t", "", "result", "s", "direct", "e", "", "", ""]])
    write("gap_review.csv", [["CG-1", "t", "s", "o", "e", "n", "4", "5", "", ""], ["CG-2", "t", "s", "o", "e", "n"] + blank(4)])
    r = score_sheets(str(tmp_path))
    assert r["extraction_field_accuracy"] == {"labelled": 2, "value": 0.5, "rows": 3}
    assert r["extraction_attribution_accuracy"]["value"] == 1.0
    assert r["gap_relevance_mean"] == {"labelled": 1, "value": 4.0, "rows": 2}
    assert r["plan_rubric_means"]["consistency"]["value"] is None       # sheet absent: not measured
