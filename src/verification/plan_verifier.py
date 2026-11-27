"""
Verification of APPROVED research proposals (Module 7).

  approved plan revision --adapter--> ProposalSpecification --VerificationEngine--> checks 1-7
                                                            + state fidelity (8)
                                                            + noise-condition consistency (9), qiskit-aer, exact
                                                            + which results rest on assumed parameters
                                                            + hardware / physical validation status
The verdict is computed by fixed rules (verdict.py); no model output can override a computational failure.
A plan that is not specified enough to simulate is held with the missing items listed, never filled in.
Every run is stored against the exact plan revision it verified.
"""

import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from ..circuit.metrics import compare_outcomes, total_variation_distance
from ..experiment_planning import load_plan
from ..experiment_planning.schema import FEASIBILITY_LABELS, ResearchProposal
from ..quantum.noise import conditions_from_parameters, noisy_simulation, state_fidelity, validate_condition
from ..schemas.proposal import CircuitSpec, ExpectedOutcome, ExtractedClaim, ProposalSpecification
from ..schemas.verification import CheckCategory, CheckStatus, VerificationCheck, VerificationReport
from ..storage import VerificationRun, session_scope
from .verdict import overall_verdict

FIDELITY_THRESHOLD = 0.99


class NotApproved(ValueError):
    pass


def plan_to_specification(p: ResearchProposal) -> ProposalSpecification:
    """Approved plan -> the engine's specification. Nothing is invented: absent circuit stays absent."""
    c = p.circuit
    assumed = [f"{x.name} = {x.value}{(' ' + x.unit) if x.unit else ''} (assumed: {x.basis})" for x in p.parameters
               if x.source == "assumption"]
    spec = dict(
        title=p.title, objective=p.research_question or p.title, hypothesis=p.hypothesis or None,
        methodology=" ".join(p.methodology) or None, quantum_system=p.quantum_system or "Ideal simulation",
        measurement_basis="computational (Z)",
        assumptions=["Ideal statevector simulation unless a noise condition is listed."] + assumed,
        missing_information=[m for m in p.missing_information], input_mode="manual", confirmed_by_researcher=True)
    if c is not None:
        n = c.num_qubits
        spec.update(num_qubits=n, initial_state="|" + "0" * n + ">",
                    circuit=CircuitSpec(num_qubits=n, gates=c.gates, initial_state="|" + "0" * n + ">"),
                    expected_outcomes=[ExpectedOutcome(basis_state=o.basis_state, expected_probability=o.expected_probability,
                                                       tolerance=o.tolerance) for o in c.expected_outcomes])
        if c.expected_outcomes:
            dist = ", ".join(f"P({o.basis_state}) = {o.expected_probability}" for o in c.expected_outcomes)
            spec["author_claims"] = [ExtractedClaim(claim_id="CLM-01", claim_type="outcome",
                                                    statement=f"Ideal measurement yields {dist}.",
                                                    source_reference=c.prediction_basis or None)]
    return ProposalSpecification(**spec)


def _check(cid: str, name: str, cat: CheckCategory, status: CheckStatus, desc: str, details: str,
           error: Optional[str] = None, **meta) -> VerificationCheck:
    return VerificationCheck(check_id=cid, name=name, category=cat, status=status, description=desc, details=details,
                             error_message=error, metadata=meta)


def extra_checks(p: ResearchProposal, report: VerificationReport) -> Tuple[List[VerificationCheck], List[Dict[str, Any]], Optional[float]]:
    """Checks 8-9 plus assumption and hardware status. Returns (checks, noise results, ideal fidelity)."""
    checks: List[VerificationCheck] = []
    c, sim = p.circuit, report.simulation
    spec = CircuitSpec(num_qubits=c.num_qubits, gates=c.gates) if c else None
    ideal_ok = sim is not None

    # --- 8. state fidelity
    fid = None
    desc8 = "Fidelity between the simulated state and the plan's expected state, when the plan states one."
    if c is None or not c.expected_state:
        checks.append(_check("CHK-FID-01", "State fidelity", CheckCategory.SIMULATION, CheckStatus.NOT_RUN, desc8,
                             "The plan states no expected state (amplitudes), so fidelity is not computed."))
    elif not ideal_ok:
        checks.append(_check("CHK-FID-01", "State fidelity", CheckCategory.SIMULATION, CheckStatus.NOT_RUN, desc8,
                             "The circuit could not be simulated."))
    else:
        import numpy as np
        psi = np.zeros(2 ** sim.num_qubits, dtype=complex)
        for comp in sim.statevector_components:
            psi[int(comp.basis_state, 2)] = complex(comp.real, comp.imag)
        try:
            fid = state_fidelity(c.expected_state, psi, sim.num_qubits)
            ok = fid >= FIDELITY_THRESHOLD
            checks.append(_check("CHK-FID-01", "State fidelity", CheckCategory.SIMULATION,
                                 CheckStatus.PASS if ok else CheckStatus.FAIL, desc8,
                                 f"Fidelity of the ideal simulated state with the expected state: {fid:.4f} "
                                 f"(threshold {FIDELITY_THRESHOLD}).",
                                 None if ok else f"The circuit does not prepare the expected state (fidelity {fid:.4f}).",
                                 fidelity=round(fid, 6)))
        except ValueError as exc:
            checks.append(_check("CHK-FID-01", "State fidelity", CheckCategory.SIMULATION, CheckStatus.FAIL, desc8,
                                 f"The expected state is invalid: {exc}.", f"Invalid expected state: {exc}"))

    # --- 9. noise conditions
    conds, notes = conditions_from_parameters(p.parameters)
    noise_results: List[Dict[str, Any]] = []
    desc9a = "The stated noise parameters are physically consistent (probabilities in [0, 1], T2 ≤ 2·T1)."
    desc9b = "Predictions under the plan's stated noise, checked by exact density-matrix simulation."
    if not conds:
        checks.append(_check("CHK-NOISE-01", "Noise parameters", CheckCategory.NOISE_CONSISTENCY, CheckStatus.NOT_RUN, desc9a,
                             "The plan states no noise parameters; only the ideal simulation was run."))
    else:
        problems = sorted({msg for cd in conds for msg in validate_condition(cd)})
        checks.append(_check("CHK-NOISE-01", "Noise parameters", CheckCategory.NOISE_CONSISTENCY,
                             CheckStatus.FAIL if problems else CheckStatus.PASS, desc9a,
                             " ".join(problems) if problems else
                             f"{len(conds)} noise condition(s) from the plan's parameters are physically consistent. " + " ".join(notes),
                             "; ".join(problems) or None))
        if not problems and spec is not None and ideal_ok:
            for cd in conds:
                probs, rho = noisy_simulation(spec, cd)
                r = {"label": cd.label, "params": {k: v for k, v in cd.model_dump().items()
                                                    if k not in ("label", "sources") and v not in (None, 0.0)},
                     "sources": cd.sources, "probabilities": probs,
                     "tvd_to_ideal": round(total_variation_distance(probs, sim.probabilities), 4)}
                if c.expected_state:
                    r["fidelity"] = round(state_fidelity(c.expected_state, rho, spec.num_qubits), 4)
                if c.expected_outcomes_noisy:
                    comps, matched, worst = compare_outcomes(c.expected_outcomes_noisy, probs)
                    r["comparison"] = [x.model_dump() for x in comps]
                    r["matched"] = matched
                elif c.expected_outcomes:
                    comps, matched, worst = compare_outcomes(c.expected_outcomes, probs)
                    r["ideal_prediction_under_noise"] = {"matched": matched, "max_error": round(worst, 4)}
                noise_results.append(r)
            judged = [r for r in noise_results if "matched" in r]
            if judged:
                bad = [r for r in judged if not r["matched"]]
                checks.append(_check("CHK-NOISE-02", "Predictions under noise", CheckCategory.NOISE_CONSISTENCY,
                                     CheckStatus.FAIL if bad else CheckStatus.PASS, desc9b,
                                     "; ".join(f"{r['label']}: {'matches' if r['matched'] else 'does not match'} the noisy "
                                               f"prediction (TVD to ideal {r['tvd_to_ideal']})" for r in judged),
                                     ("The plan's noisy predictions disagree with simulation under: "
                                      + ", ".join(r["label"] for r in bad)) if bad else None))
            else:
                checks.append(_check("CHK-NOISE-02", "Predictions under noise", CheckCategory.NOISE_CONSISTENCY,
                                     CheckStatus.NOT_RUN, desc9b,
                                     "The plan gives no predictions under noise, so there is nothing to check; the effect of the "
                                     "stated noise is reported: " + "; ".join(
                                         f"{r['label']}: TVD to ideal {r['tvd_to_ideal']}"
                                         + (f", ideal prediction {'still holds' if r['ideal_prediction_under_noise']['matched'] else 'no longer holds'}"
                                            if 'ideal_prediction_under_noise' in r else "") for r in noise_results)))
        elif not problems:
            checks.append(_check("CHK-NOISE-02", "Predictions under noise", CheckCategory.NOISE_CONSISTENCY, CheckStatus.NOT_RUN,
                                 desc9b, "No circuit could be simulated, so the noise conditions were not run."))

    # --- assumptions the results rest on
    used = [x for x in p.parameters if x.source == "assumption"]
    noise_assumed = [s for cd in conds[:1] for s in cd.sources.values() if "(assumption)" in s or "default assumption" in s]
    pred_basis = (c.prediction_basis if c else "") or ""
    desc_a = "Lists the assumed (not literature- or researcher-sourced) values that the verification depends on."
    if noise_assumed:
        checks.append(_check("CHK-ASSUME-01", "Assumptions behind the result", CheckCategory.ASSUMPTIONS, CheckStatus.INCONCLUSIVE,
                             desc_a, "The noise simulation uses assumed values: " + "; ".join(noise_assumed) + ".",
                             "Noise results rest on assumed parameters: " + "; ".join(noise_assumed)))
    else:
        checks.append(_check("CHK-ASSUME-01", "Assumptions behind the result", CheckCategory.ASSUMPTIONS, CheckStatus.PASS, desc_a,
                             ("No simulated value rests on an assumption." if conds else "Only the ideal circuit was simulated.")
                             + (f" The plan also assumes: {'; '.join(x.name + ' = ' + x.value for x in used)} (not used in simulation)."
                                if used else "") + (f" Predictions are based on: {pred_basis}" if pred_basis else "")))

    # --- what simulation cannot establish
    hw = {"simulatable": "The plan is simulatable; this verdict covers the ideal and modelled-noise simulation only, "
                         "not behaviour on physical hardware.",
          "needs_hardware": "This plan requires quantum hardware. The simulation checks the circuit and predictions only; "
                            "hardware validation has not been performed.",
          "needs_physical_validation": "This plan requires additional physical validation that circuit simulation cannot provide; "
                                       "it has not been performed."}[p.feasibility_class]
    checks.append(_check("CHK-HW-01", "Hardware and physical validation", CheckCategory.PHYSICAL_FEASIBILITY, CheckStatus.NOT_RUN,
                         f"Feasibility: {FEASIBILITY_LABELS[p.feasibility_class]}.", hw))
    return checks, noise_results, fid


def verify_plan(lineage: int, engine, db_url: Optional[str] = None) -> Dict[str, Any]:
    plan = load_plan(lineage, db_url)
    if plan is None:
        raise NotApproved(f"No plan {lineage}")
    if plan["status"] != "approved":
        raise NotApproved("Only an approved plan can be verified. Approve the current revision first.")
    p: ResearchProposal = plan["proposal"]
    spec = plan_to_specification(p)
    report = engine.verify_proposal(spec)
    extra, noise, fid = extra_checks(p, report)
    report.checks.extend(extra)
    report.fidelity_score = fid
    report.summary_counts = {k: sum(1 for c in report.checks if c.status.value == k)
                             for k in ("PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN")}
    for c in extra:
        if c.status == CheckStatus.FAIL and c.error_message:
            report.detected_inconsistencies.append(c.error_message)
    verdict = overall_verdict(report)
    if verdict == "fail" and "FAILED" not in report.overall_summary:
        report.overall_summary = ("VERIFICATION STATUS: FAILED. " + "; ".join(c.error_message for c in extra
                                                                             if c.status == CheckStatus.FAIL and c.error_message))
    import qiskit
    try:
        import qiskit_aer
        aer = f" + qiskit-aer {qiskit_aer.__version__} density matrix" if noise else ""
    except ImportError:              # pragma: no cover
        aer = ""
    simulator = f"qiskit {qiskit.__version__} statevector{aer}"
    report.software_environment["python_version"] = platform.python_version()
    with session_scope(db_url) as s:
        run = VerificationRun(plan_id=plan["record_id"], verdict=verdict, simulator=simulator,
                              report={"report": report.model_dump(mode="json"), "noise": noise, "revision": plan["revision"],
                                      "lineage": lineage, "spec": spec.model_dump(mode="json"),
                                      "feasibility": p.feasibility_class},
                              ran_at=datetime.now(timezone.utc))
        s.add(run)
        s.flush()
        run_id = run.id
    return {"run_id": run_id, "verdict": verdict, "report": report, "noise": noise, "revision": plan["revision"],
            "simulator": simulator}


def load_runs(lineage: int, db_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Verification runs of every revision of a plan, newest first."""
    with session_scope(db_url) as s:
        runs = [r for r in s.scalars(select(VerificationRun).order_by(VerificationRun.ran_at.desc(), VerificationRun.id.desc()))
                if (r.report or {}).get("lineage") == lineage]
        return [{"run_id": r.id, "verdict": r.verdict, "simulator": r.simulator, "revision": r.report.get("revision"),
                 "at": r.ran_at.isoformat() if r.ran_at else None, "report": VerificationReport(**r.report["report"]),
                 "spec": ProposalSpecification(**r.report["spec"]), "noise": r.report.get("noise", []),
                 "feasibility": r.report.get("feasibility")} for r in runs]
