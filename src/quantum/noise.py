"""
Noise-model simulation for plan verification (Module 7, check 9).

Deterministic: exact density-matrix simulation with qiskit-aer (no shots), and readout error applied analytically,
so the same plan and parameters always give the same numbers. Noise parameters come only from the plan's own
parameters, each carrying its source (literature, researcher or assumption); nothing is defaulted silently.
A stated range ("0.001-0.05") is simulated at both ends.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..schemas.proposal import CircuitSpec

ONE_Q = ["h", "x", "y", "z", "s", "t", "sdg", "tdg", "rx", "ry", "rz", "id"]
TWO_Q = ["cx", "cz", "swap"]
DEFAULT_GATE_TIME_NS = {"1q": 50.0, "2q": 300.0}     # used only with T1/T2, and reported as an assumption


class NoiseCondition(BaseModel):
    label: str
    depolarizing_1q: float = 0.0
    depolarizing_2q: float = 0.0
    readout: float = 0.0
    t1_us: Optional[float] = None
    t2_us: Optional[float] = None
    gate_time_1q_ns: Optional[float] = None
    gate_time_2q_ns: Optional[float] = None
    sources: Dict[str, str] = Field(default_factory=dict, description="Field -> 'name = value (source)' it came from.")

    @property
    def is_noiseless(self) -> bool:
        return not (self.depolarizing_1q or self.depolarizing_2q or self.readout or self.t1_us or self.t2_us)


def validate_condition(c: NoiseCondition) -> List[str]:
    """Physical consistency of the stated noise; any problem fails verification."""
    out = []
    for f in ("depolarizing_1q", "depolarizing_2q", "readout"):
        v = getattr(c, f)
        if not 0.0 <= v <= 1.0:
            out.append(f"{f.replace('_', ' ')} = {v} is not a probability in [0, 1].")
    for f in ("t1_us", "t2_us", "gate_time_1q_ns", "gate_time_2q_ns"):
        v = getattr(c, f)
        if v is not None and v <= 0:
            out.append(f"{f} = {v} must be positive.")
    if c.t1_us and c.t2_us and c.t2_us > 2 * c.t1_us + 1e-9:
        out.append(f"T2 = {c.t2_us} µs exceeds 2·T1 = {2 * c.t1_us} µs, which is physically impossible.")
    return out


# ------------------------------------------------------------------ parameters
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_values(value: str, unit: str = "") -> List[float]:
    """'0.01' -> [0.01]; '1%' -> [0.01]; '0.001-0.05' / '0.1 to 1' -> [low, high]."""
    text = f"{value} {unit}"
    nums = [float(x) for x in re.findall(_NUM, value.replace("–", "-").replace(" to ", " - "))]
    nums = [abs(x) for x in nums]                       # "0.001-0.05" parses the dash as a sign
    if not nums:
        return []
    if "%" in text:
        nums = [x / 100 for x in nums]
    return [min(nums), max(nums)] if len(nums) >= 2 else nums


def _time_scale(unit: str, default: str) -> float:
    u = (unit or default).lower()
    return {"ns": 1e-3, "us": 1.0, "µs": 1.0, "μs": 1.0, "microseconds": 1.0, "ms": 1e3, "s": 1e6}.get(u.strip(), 1.0)


def conditions_from_parameters(params) -> Tuple[List[NoiseCondition], List[str]]:
    """
    Map plan parameters onto noise fields by name. Returns (conditions, notes). No noise parameters -> [].
    With ranges, two conditions: every parameter at its low end, and every parameter at its high end.
    """
    fields: Dict[str, Tuple[List[float], str]] = {}
    rank: Dict[str, int] = {}
    notes: List[str] = []
    precedence = {"researcher": 3, "literature": 2, "gap_evidence": 2, "assumption": 1}

    def put(field: str, vals: List[float], src: str, source: str):
        """Two parameters for the same quantity: the researcher's beats the literature's beats an assumption."""
        r = precedence.get(source, 0)
        if field in fields:
            keep_new = r >= rank[field]
            notes.append(f"Two parameters set {field.replace('_', ' ')}: using “{(src if keep_new else fields[field][1])}”, "
                         f"ignoring “{(fields[field][1] if keep_new else src)}”.")
            if not keep_new:
                return
        fields[field], rank[field] = (vals, src), r
    for p in params:
        n = p.name.lower()
        vals = parse_values(p.value, p.unit)
        if not vals:
            continue
        src = f"{p.name} = {p.value}{(' ' + p.unit) if p.unit else ''} ({p.source})"
        two = any(k in n for k in ("two-qubit", "2q", "2-qubit", "cx", "cnot", "entangling"))
        one = any(k in n for k in ("single-qubit", "1q", "1-qubit", "single qubit"))
        if "depolari" in n or "error rate" in n or "gate error" in n:
            if "depolari" not in n:
                notes.append(f"“{p.name}” is interpreted as a depolarizing probability per gate.")
            targets = ["depolarizing_2q"] if two else ["depolarizing_1q"] if one else ["depolarizing_1q", "depolarizing_2q"]
            for t in targets:
                put(t, vals, src, p.source)
        elif "readout" in n or "measurement error" in n or "spam" in n:
            put("readout", vals, src, p.source)
        elif re.search(r"\bt1\b", n) or "relaxation time" in n:
            put("t1_us", [v * _time_scale(p.unit, "us") for v in vals], src, p.source)
        elif re.search(r"\bt2\b", n) or "dephasing time" in n or "coherence time" in n:
            put("t2_us", [v * _time_scale(p.unit, "us") for v in vals], src, p.source)
        elif "gate time" in n or "gate duration" in n:
            key = "gate_time_2q_ns" if two else "gate_time_1q_ns"
            put(key, [v * _time_scale(p.unit, "ns") * 1e3 for v in vals], src, p.source)     # -> ns
    if not fields:
        return [], notes
    if ("t1_us" in fields or "t2_us" in fields):
        for k, kind in (("gate_time_1q_ns", "1q"), ("gate_time_2q_ns", "2q")):
            if k not in fields:
                fields[k] = ([DEFAULT_GATE_TIME_NS[kind]], f"{kind} gate time = {DEFAULT_GATE_TIME_NS[kind]} ns (default assumption)")
                notes.append(f"No {kind} gate time was stated; {DEFAULT_GATE_TIME_NS[kind]} ns is assumed for relaxation.")
    ranged = any(len(v) > 1 for v, _ in fields.values())
    ends = [("low end of every stated range", 0), ("high end of every stated range", -1)] if ranged else [("as stated", 0)]
    conds = []
    for label, i in ends:
        kw = {k: v[i] for k, (v, _) in fields.items()}
        conds.append(NoiseCondition(label=label, sources={k: s for k, (_, s) in fields.items()}, **kw))
    return conds, notes


# ------------------------------------------------------------------ simulation
def to_qiskit(spec: CircuitSpec):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(spec.num_qubits)
    for g in spec.gates:
        name, q, prm = g.name.upper(), g.target_qubits, g.params or []
        method = {"CNOT": "cx", "ID": "id", "I": "id"}.get(name, name.lower())
        if method in ("rx", "ry", "rz"):
            getattr(qc, method)(prm[0], q[0])
        else:
            getattr(qc, method)(*q)
    return qc


def _readout(probs: np.ndarray, n: int, p: float) -> np.ndarray:
    """Independent symmetric bit-flip on every measured qubit (exact, no sampling)."""
    if p <= 0:
        return probs
    flip = np.array([[1 - p, p], [p, 1 - p]])
    t = probs.reshape([2] * n)               # axis k <-> qubit n-1-k (big-endian bitstring order)
    for axis in range(n):
        t = np.moveaxis(np.tensordot(flip, t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


def noisy_simulation(spec: CircuitSpec, c: NoiseCondition) -> Tuple[Dict[str, float], np.ndarray]:
    """(probabilities by bitstring in Qiskit order, density matrix before readout) under the condition."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

    model = NoiseModel()
    errs = {1: None, 2: None}
    if c.depolarizing_1q:
        errs[1] = depolarizing_error(c.depolarizing_1q, 1)
    if c.depolarizing_2q:
        errs[2] = depolarizing_error(c.depolarizing_2q, 2)
    if c.t1_us or c.t2_us:
        t1 = (c.t1_us or 1e9) * 1e3                    # ns
        t2 = min((c.t2_us or 1e9) * 1e3, 2 * t1)
        r1 = thermal_relaxation_error(t1, t2, c.gate_time_1q_ns or DEFAULT_GATE_TIME_NS["1q"])
        r2 = thermal_relaxation_error(t1, t2, c.gate_time_2q_ns or DEFAULT_GATE_TIME_NS["2q"])
        r2 = r2.expand(r2)
        errs[1] = r1 if errs[1] is None else errs[1].compose(r1)
        errs[2] = r2 if errs[2] is None else errs[2].compose(r2)
    if errs[1] is not None:
        model.add_all_qubit_quantum_error(errs[1], ONE_Q)
    if errs[2] is not None:
        model.add_all_qubit_quantum_error(errs[2], TWO_Q)

    qc = to_qiskit(spec)
    qc.save_density_matrix(label="rho")
    sim = AerSimulator(method="density_matrix", noise_model=model if not model.is_ideal() else None)
    data = sim.run(qc, basis_gates=sorted(set(ONE_Q + TWO_Q))).result().data(0)
    rho = np.asarray(data["rho"])
    n = spec.num_qubits
    probs = _readout(np.clip(np.real(np.diag(rho)), 0, None), n, c.readout)
    probs = probs / probs.sum()
    return {format(i, f"0{n}b"): float(round(v, 10)) for i, v in enumerate(probs) if v > 1e-12}, rho


def state_fidelity(expected: Dict[str, List[float]], rho_or_psi: np.ndarray, n: int) -> float:
    """F = <psi|rho|psi> for an expected pure state given as {bitstring: [re, im]} (normalised here)."""
    psi = np.zeros(2 ** n, dtype=complex)
    for b, (re_, im) in expected.items():
        psi[int(b, 2)] = complex(re_, im)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("expected state has zero norm")
    psi /= norm
    m = rho_or_psi if rho_or_psi.ndim == 2 else np.outer(rho_or_psi, rho_or_psi.conj())
    return float(min(1.0, max(0.0, np.real(psi.conj() @ m @ psi))))     # clamp float round-off
