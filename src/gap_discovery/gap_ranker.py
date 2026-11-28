"""
Gap ranking (Module 5): eight separate dimensions, each 0-1 by a stated rubric with a one-line justification.
There is no hidden composite: the weighted total uses weights the researcher sees and can change.
"""

from typing import Dict, List, Optional, Tuple

from .schema import CLASSIFICATION_LABELS, Gap

RANK_DIMENSIONS: List[Tuple[str, str]] = [
    ("novelty", "Novelty"), ("significance", "Significance"), ("evidence", "Evidence strength"),
    ("feasibility", "Feasibility"), ("testability", "Testability"), ("tools", "Tool availability"),
    ("contribution", "Potential contribution"), ("coverage", "Literature coverage"),
]
DEFAULT_WEIGHTS: Dict[str, float] = {"novelty": 2.0, "significance": 1.0, "evidence": 1.5, "feasibility": 1.0,
                                     "testability": 1.0, "tools": 0.5, "contribution": 1.0, "coverage": 1.0}

# Novelty comes from the investigation's classification, never from similarity.
NOVELTY_BY_CLASS = {"potentially_unexplored": 0.9, "partially_investigated": 0.6, "conflicting_findings": 0.7,
                    "insufficient_coverage": 0.3, "already_investigated": 0.1}
SIMULABLE_METHODS = {"quantum error mitigation", "quantum error correction", "variational algorithms", "dynamical decoupling",
                     "benchmarking", "tomography", "quantum teleportation", "quantum machine learning"}
SIMULABLE_PHENOMENA = {"entanglement", "measurement-induced phase transitions", "barren plateaus"}
SIMULATOR_QUBITS = 20          # qiskit-aer statevector on a laptop
EVIDENCE_WEIGHT = {"direct": 1.0, "interpretation": 0.7, "inference": 0.4}


def _vals(g: Gap, d: str) -> List[str]:
    return g.dimensions[d].values if d in g.dimensions else []


def score_gap(g: Gap, assessment: Optional[dict]) -> Dict[str, dict]:
    s: Dict[str, dict] = {}
    supporting = [e for e in g.evidence if e.role == "supporting"]
    papers = len({e.publication_id for e in g.evidence if e.publication_id})

    if assessment:
        cls = assessment["classification"]
        s["novelty"] = {"score": NOVELTY_BY_CLASS.get(cls, 0.3),
                        "why": f"{CLASSIFICATION_LABELS.get(cls, cls)} (confidence {assessment['confidence']:.2f})."}
    else:
        s["novelty"] = {"score": 0.3, "why": "Not investigated yet; run a novelty check."}

    raised_by = len({e.publication_id for e in supporting})
    sig = min(1.0, 0.3 + 0.2 * raised_by + (0.1 if g.origin in ("unresolved_problem", "future_work") else 0.0))
    s["significance"] = {"score": round(sig, 2), "why": (f"Raised by {raised_by} paper{'s' if raised_by != 1 else ''}"
                                                         if raised_by else "Inferred from corpus coverage, not stated by authors")
                         + ("; authors call it open." if g.origin in ("unresolved_problem", "future_work") else ".")}

    if g.evidence:
        q = sum(EVIDENCE_WEIGHT[e.kind] for e in g.evidence) / len(g.evidence)
        ev = q * min(1.0, 0.6 + 0.2 * papers)
        kinds = sorted({e.kind for e in g.evidence})
        s["evidence"] = {"score": round(ev, 2), "why": f"{len(g.evidence)} item(s) from {papers} paper(s); kinds: {', '.join(kinds)}."}
    else:
        s["evidence"] = {"score": 0.0, "why": "No evidence recorded."}

    qubits = [int(v) for v in _vals(g, "qubit_scale") if str(v).isdigit()]
    platforms = _vals(g, "platform")
    if qubits and max(qubits) > SIMULATOR_QUBITS:
        feas, why = 0.3, f"Needs {max(qubits)} qubits; beyond a {SIMULATOR_QUBITS}-qubit simulator."
    elif platforms:
        feas, why = 0.6, f"Can be simulated, but the question concerns {', '.join(platforms)} hardware."
    else:
        feas, why = 0.85, "Small enough to study on a simulator."
    s["feasibility"] = {"score": feas, "why": why}

    metric = _vals(g, "metric")
    test = 0.8 if metric else (0.5 if qubits or _vals(g, "noise") else 0.35)
    s["testability"] = {"score": test, "why": f"Measurable through {', '.join(metric)}." if metric else
                        ("A measurable condition (scale or noise) is named, but no metric." if test == 0.5 else
                         "No metric or measurable condition named yet; the plan must define one.")}

    methods, phen = set(_vals(g, "method")), set(_vals(g, "phenomenon"))
    tool = 1.0 if methods & SIMULABLE_METHODS or phen & SIMULABLE_PHENOMENA else 0.4
    s["tools"] = {"score": tool, "why": "Supported by the Qiskit simulator and noise models in this project." if tool == 1.0
                  else "No method here maps onto the project's simulator tooling."}

    contrib = {"future_work": 0.8, "unresolved_problem": 0.8, "unexplored_combination": 0.6,
               "stated_limitation": 0.5}.get(g.origin, 0.5) + (0.1 if raised_by > 1 else 0.0)
    s["contribution"] = {"score": round(min(1.0, contrib), 2),
                         "why": {"future_work": "Authors name it as the next step.", "unresolved_problem": "Authors say it is open.",
                                 "unexplored_combination": "Would fill an empty cell of the corpus coverage.",
                                 "stated_limitation": "Would lift a limitation of published work."}.get(g.origin, "")
                                + (" Raised by several papers." if raised_by > 1 else "")}

    if assessment:
        cov = assessment["confidence_basis"].get("coverage", 0.0)
        c = assessment["coverage"]
        s["coverage"] = {"score": round(cov, 2), "why": f"{len(c['sources_ok'])} of {len(c['sources_attempted'])} sources answered, "
                                                        f"{len(c['queries'])} queries, {c['screened']} papers screened."}
    else:
        s["coverage"] = {"score": 0.0, "why": "No literature search yet."}
    return s


def weighted_total(scores: Dict[str, dict], weights: Optional[Dict[str, float]] = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    tot = sum(max(0.0, w[k]) for k, _ in RANK_DIMENSIONS)
    return round(sum(max(0.0, w[k]) * scores[k]["score"] for k, _ in RANK_DIMENSIONS) / tot, 3) if tot else 0.0
