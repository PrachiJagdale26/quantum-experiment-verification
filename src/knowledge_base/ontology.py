"""
Curated quantum-physics concept map (deterministic, reviewable).

Each canonical concept has a category and the surface forms papers use for it. It is used to:
  - tag papers with research dimensions from their own wording (no LLM involved);
  - expand search queries with equivalent terminology (decoherence ~ dephasing ~ T2 ~ coherence loss);
  - compare research dimensions across papers.
Related concepts are linked but never treated as identical: "related" is a hint for retrieval, not
evidence that two papers study the same question.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

# canonical: (category, [surface forms], [related canonical concepts])
CONCEPTS: Dict[str, Tuple[str, List[str], List[str]]] = {
    # --- hardware platforms
    "superconducting qubits": ("platform", ["superconducting qubit", "superconducting processor", "superconducting circuit",
                                            "transmon", "fluxonium", "superconducting quantum"], []),
    "trapped ions": ("platform", ["trapped ion", "trapped-ion", "ion trap", "ion chain", "paul trap"], []),
    "neutral atoms": ("platform", ["neutral atom", "neutral-atom", "rydberg atom", "optical tweezer", "atom array"], []),
    "photonic qubits": ("platform", ["photonic", "linear optical", "optical quantum computing", "integrated photonics"], []),
    "spin qubits": ("platform", ["spin qubit", "silicon qubit", "quantum dot", "silicon spin"], []),
    "nv centers": ("platform", ["nitrogen-vacancy", "nv center", "nv centre", "color center", "colour centre"], []),
    "topological qubits": ("platform", ["topological qubit", "majorana"], []),
    # --- noise and decoherence
    "decoherence": ("noise", ["decoherence", "dephasing", "coherence time", "loss of coherence", "coherence loss", "t2 time",
                              "t_2", "quantum state degradation"], ["amplitude damping", "environmental noise"]),
    "amplitude damping": ("noise", ["amplitude damping", "energy relaxation", "t1 time", "t_1", "relaxation time"], ["decoherence"]),
    "environmental noise": ("noise", ["environmental noise", "open quantum system", "system-environment", "bath coupling",
                                      "markovian noise", "lindblad"], ["decoherence", "non-markovian noise"]),
    "non-markovian noise": ("noise", ["non-markovian", "memory effects", "correlated noise"], ["environmental noise"]),
    "depolarizing noise": ("noise", ["depolarizing", "depolarising", "pauli noise", "stochastic pauli"], []),
    "readout error": ("noise", ["readout error", "measurement error", "spam error", "state preparation and measurement"], []),
    "crosstalk": ("noise", ["crosstalk", "cross-talk", "zz coupling", "spectator"], []),
    "leakage": ("noise", ["leakage"], []),
    "coherent errors": ("noise", ["coherent error", "over-rotation", "miscalibration", "control error"], []),
    "hardware noise (unspecified)": ("noise", ["noisy quantum", "noisy intermediate-scale", "nisq", "hardware noise",
                                               "noisy hardware", "noisy device", "noisy processor", "gate noise"],
                                     ["decoherence", "depolarizing noise"]),
    # --- methods and techniques
    "quantum error mitigation": ("method", ["error mitigation", "zero-noise extrapolation", "zero noise extrapolation", "zne",
                                            "probabilistic error cancellation", "clifford data regression", "symmetry verification",
                                            "virtual distillation"], ["quantum error correction", "readout error"]),
    "quantum error correction": ("method", ["error correction", "error-correcting code", "surface code", "stabilizer code",
                                            "logical qubit", "color code", "bosonic code", "fault-tolerant", "fault tolerance",
                                            "decoder"], ["quantum error mitigation"]),
    "variational algorithms": ("method", ["variational quantum", "vqe", "variational eigensolver", "qaoa",
                                          "approximate optimization algorithm", "parameterized quantum circuit",
                                          "parametrized quantum circuit", "ansatz"], ["quantum machine learning", "barren plateaus"]),
    "quantum machine learning": ("method", ["quantum machine learning", "quantum neural network", "quantum kernel",
                                            "quantum classifier", "qml"], ["variational algorithms"]),
    "dynamical decoupling": ("method", ["dynamical decoupling", "spin echo", "hahn echo"], ["decoherence"]),
    "benchmarking": ("method", ["randomized benchmarking", "cross-entropy benchmarking", "quantum volume", "benchmarking"], []),
    "tomography": ("method", ["state tomography", "process tomography", "classical shadow", "shadow tomography", "tomography"], []),
    "quantum teleportation": ("method", ["teleportation"], ["entanglement"]),
    "quantum key distribution": ("method", ["quantum key distribution", "qkd", "bb84", "e91"], ["quantum communication"]),
    "quantum communication": ("method", ["quantum communication", "quantum network", "quantum repeater", "quantum internet"],
                              ["quantum key distribution", "entanglement"]),
    "circuit compilation": ("method", ["circuit optimization", "circuit optimisation", "transpil", "gate synthesis",
                                       "circuit compilation", "qubit routing", "qubit mapping"], []),
    "state preparation": ("method", ["state preparation", "preparing the state", "ghz state", "bell state", "w state",
                                     "cluster state", "graph state"], ["entanglement"]),
    "quantum simulation": ("method", ["quantum simulation", "hamiltonian simulation", "trotter", "digital simulation"], []),
    # --- phenomena
    "entanglement": ("phenomenon", ["entanglement", "entangled", "bell pair", "epr pair", "ghz", "bell state"],
                     ["decoherence", "quantum teleportation"]),
    "measurement-induced phase transitions": ("phenomenon", ["measurement-induced", "monitored circuit", "monitored quantum"],
                                              ["entanglement"]),
    "barren plateaus": ("phenomenon", ["barren plateau"], ["variational algorithms"]),
    "quantum advantage": ("phenomenon", ["quantum advantage", "quantum supremacy", "beyond-classical"], []),
    # --- metrics
    "fidelity": ("metric", ["fidelity", "infidelity"], []),
    "logical error rate": ("metric", ["logical error rate", "logical error"], ["quantum error correction"]),
    "gate error": ("metric", ["gate error", "two-qubit gate fidelity", "gate fidelity"], ["fidelity"]),
    "energy accuracy": ("metric", ["chemical accuracy", "ground-state energy", "energy error"], ["variational algorithms"]),
    "total variation distance": ("metric", ["total variation distance", "tvd"], []),
}

CATEGORIES = ("platform", "noise", "method", "phenomenon", "metric")


@dataclass(frozen=True)
class ConceptHit:
    concept: str
    category: str
    surface: str
    start: int
    end: int


@lru_cache(maxsize=1)
def _patterns() -> List[Tuple[re.Pattern, str, str]]:
    pats = []
    for concept, (cat, forms, _) in CONCEPTS.items():
        for form in sorted(set(forms + [concept]), key=len, reverse=True):
            # word-start boundary; allow plural / suffix only for multi-letter stems
            core = re.escape(form)
            suffix = r"" if len(form) <= 4 else r"\w*"
            pats.append((re.compile(rf"(?<![A-Za-z0-9]){core}{suffix}(?![A-Za-z0-9])", re.I), concept, cat))
    return pats


def find_concepts(text: str) -> List[ConceptHit]:
    """All concept mentions in a text, with spans (overlapping shorter matches of the same concept removed)."""
    hits: List[ConceptHit] = []
    for pat, concept, cat in _patterns():
        for m in pat.finditer(text or ""):
            hits.append(ConceptHit(concept, cat, m.group(0), m.start(), m.end()))
    hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
    kept: List[ConceptHit] = []
    for h in hits:
        if any(k.concept == h.concept and k.start <= h.start < k.end for k in kept):
            continue
        kept.append(h)
    return kept


def concepts_by_category(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    for h in find_concepts(text):
        if h.concept not in out[h.category]:
            out[h.category].append(h.concept)
    return out


def synonyms(concept: str) -> List[str]:
    return list(CONCEPTS.get(concept, ("", [], []))[1])


def related(concept: str) -> List[str]:
    return list(CONCEPTS.get(concept, ("", [], []))[2])


def expand_query(query: str, max_variants: int = 6) -> List[Tuple[str, str]]:
    """
    Deterministic query variants: [(query, kind)], kind = exact | synonym | related.
    Each recognised concept in the query is swapped for its other surface forms (synonym) and,
    separately, for related concepts (related). The original query is always first.
    """
    variants: List[Tuple[str, str]] = [(query, "exact")]
    seen = {query.lower()}
    for h in find_concepts(query):
        for form in synonyms(h.concept)[:4]:
            v = (query[:h.start] + form + query[h.end:]).strip()
            if v.lower() not in seen:
                variants.append((v, "synonym"))
                seen.add(v.lower())
        for rel in related(h.concept)[:2]:
            v = (query[:h.start] + rel + query[h.end:]).strip()
            if v.lower() not in seen:
                variants.append((v, "related"))
                seen.add(v.lower())
    return variants[:max_variants]
