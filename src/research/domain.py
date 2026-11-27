"""
Quantum-domain vocabulary shared by the gap finder and the experiment planner:
circuit families, hardware platforms, and analytic circuit templates whose ideal
outcome distributions are known from theory (not from the simulator).
"""

import re
from typing import Dict, List, Optional, Tuple

from ..schemas.proposal import ExpectedOutcome, GateOperation
from ..schemas.research_paper import QuantumExperimentRecord, ResearchPaperAnalysis

FAMILY_PATTERNS: Dict[str, str] = {
    "ghz": r"\bghz\b|greenberger|tripartite",
    "bell": r"\bbell\b|\bepr\b|phi\s*\+|maximally entangled pair",
    "vqe": r"\bvqe\b|variational quantum eigensolver|variational eigen",
    "qaoa": r"\bqaoa\b|approximate optimi[sz]ation algorithm",
    "qft": r"\bqft\b|quantum fourier",
    "grover": r"\bgrover\b|amplitude amplification",
}
FAMILY_LABELS = {
    "ghz": "GHZ state", "bell": "Bell state", "vqe": "VQE reference state",
    "qaoa": "QAOA", "qft": "Quantum Fourier transform", "grover": "Grover search",
}
TEMPLATE_FAMILIES = {"bell", "ghz", "vqe"}

SYSTEM_PATTERNS: Dict[str, str] = {
    "Superconducting Transmon": r"superconducting|transmon",
    "Trapped Ion": r"trapped[- ]ions?|ion[- ]trap|\bions?\b chain",
    "Neutral Atom": r"neutral[- ]atoms?|rydberg",
    "Photonic": r"photonic",
    "Silicon Spin": r"silicon|spin qubit",
}

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def families_in(text: str) -> List[str]:
    return [f for f, pat in FAMILY_PATTERNS.items() if re.search(pat, text or "", re.I)]


def systems_in(text: str) -> List[str]:
    hits = []
    for name, pat in SYSTEM_PATTERNS.items():
        m = re.search(pat, text or "", re.I)
        if m:
            hits.append((m.start(), name))
    return [n for _, n in sorted(hits)]


def qubits_in(text: str) -> List[int]:
    nums = [int(n) for n in re.findall(r"\b(\d{1,2})\s*(?:or more\s*)?-?\s*qubits?\b", text or "", re.I)]
    nums += [_WORD_NUM[w.lower()] for w in re.findall(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)[- ]qubits?\b", text or "", re.I)]
    return nums


def experiment_family(exp: QuantumExperimentRecord, paper: Optional[ResearchPaperAnalysis] = None) -> Optional[str]:
    names = [g.name.upper() for g in exp.gates]
    if names and names[0] == "H" and all(n in ("CX", "CNOT") for n in names[1:]) and len(names) >= 2:
        return "bell" if (exp.num_qubits or len(names)) <= 2 else "ghz"
    if names and names[0] == "X" and "CX" in names:
        return "vqe"
    fams = families_in(exp.title or "")
    if fams:
        return fams[0]
    if paper is not None:
        fams = families_in(paper.title)
        return fams[0] if fams else None
    return None


def paper_family(paper: ResearchPaperAnalysis) -> Optional[str]:
    for exp in paper.experiments:
        fam = experiment_family(exp, paper)
        if fam:
            return fam
    fams = families_in(paper.title + " " + (paper.abstract or ""))
    return fams[0] if fams else None


def paper_system(paper: ResearchPaperAnalysis) -> Optional[str]:
    found = systems_in(paper.title + " " + (paper.abstract or ""))
    if found:
        return found[0]
    real = [s for s in paper.quantum_systems_discussed if s != "Ideal Statevector Simulation"]
    return real[0] if real else None


def template_circuit(family: str, n: int) -> Tuple[int, List[GateOperation], List[ExpectedOutcome], str]:
    """Analytic circuit + ideal outcome distribution from textbook theory."""
    if family == "ghz":
        n = max(3, min(n, 10))
        gates = [GateOperation(name="H", target_qubits=[0])] + [
            GateOperation(name="CX", target_qubits=[i, i + 1]) for i in range(n - 1)]
        outcomes = [ExpectedOutcome(basis_state="0" * n, expected_probability=0.5),
                    ExpectedOutcome(basis_state="1" * n, expected_probability=0.5)]
        why = f"An n-qubit GHZ state (|0…0⟩ + |1…1⟩)/√2 yields 0…0 and 1…1 with probability 1/2 each."
        return n, gates, outcomes, why
    if family == "vqe":
        gates = [GateOperation(name="X", target_qubits=[0]), GateOperation(name="CX", target_qubits=[0, 1])]
        outcomes = [ExpectedOutcome(basis_state="11", expected_probability=1.0)]
        why = "X on q0 then CNOT(q0→q1) maps |00⟩ to |11⟩ deterministically (Hartree–Fock-style reference)."
        return 2, gates, outcomes, why
    # Bell state is the default two-qubit entangling benchmark.
    gates = [GateOperation(name="H", target_qubits=[0]), GateOperation(name="CX", target_qubits=[0, 1])]
    outcomes = [ExpectedOutcome(basis_state="00", expected_probability=0.5),
                ExpectedOutcome(basis_state="11", expected_probability=0.5)]
    why = "H then CNOT on |00⟩ gives |Φ+⟩ = (|00⟩ + |11⟩)/√2, so 00 and 11 each occur with probability 1/2."
    return 2, gates, outcomes, why
