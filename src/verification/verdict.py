"""
Single reviewer-facing verdict derived from a VerificationReport.

- fail: any check failed.
- pass: the four computational checks (qubit count, circuit, normalization,
        outcome distribution) all passed. Literature grounding and the hardware
        disclaimer inform the reviewer but do not block a pass.
- hold: anything else (missing parameters, nothing to simulate).
"""

from ..schemas.verification import CheckStatus, VerificationReport

CORE_CHECKS = ("CHK-STRUC-01", "CHK-CIRC-01", "CHK-SIM-01", "CHK-DIST-01")


def overall_verdict(report: VerificationReport) -> str:
    by_id = {c.check_id: c for c in report.checks}
    if any(c.status == CheckStatus.FAIL for c in report.checks):
        return "fail"
    if all(by_id.get(cid) is not None and by_id[cid].status == CheckStatus.PASS for cid in CORE_CHECKS):
        return "pass"
    return "hold"
