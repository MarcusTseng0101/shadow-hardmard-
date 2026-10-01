"""Gate-level controlled time evolution (second-order Trotter) for the Hadamard test.

The exact path in experiment.py appends one 16x16 UnitaryGate. Hardware cannot run that, and a
noise model needs real gates to attach errors to, so this module builds

    controlled-[ S2(delta) ]^steps ,   S2(delta) = prod_k e^{-i c_k P_k delta/2} * reversed(...)

from H, S, S^dag, RZ and CX only.

Design choices that matter:
  * A FIXED step delta for every sample time (t_k = k * dt, dt = s * delta). Then
    U(t_k) = S2(delta)^(s k) exactly, the signal is an exact sum of exponentials of the Floquet
    quasi-energies of S2, and Trotter error separates cleanly from shot / gate noise.
  * Terms are sorted by qubit support, so XX, YY, ZZ on the same bond are adjacent. They commute,
    their product is the exact bond exponential, and every factor conserves M_z = sum Z_i:
    the symmetry labels stay EXACT under Trotterization (tested).
  * Controlled e^{-i theta Z_t} = CRZ(2 theta) is decomposed as RZ(theta)-CX-RZ(-theta)-CX, so it
    is exact including phase - the relative phase between ancilla branches is the whole signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm


@dataclass(frozen=True)
class Trotter:
    """Second-order Trotter evolution with a fixed step `delta`."""
    delta: float

    def steps(self, t: float) -> int:
        s = round(t / self.delta)
        if not math.isclose(s * self.delta, t, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"t = {t} is not a multiple of the Trotter step {self.delta}")
        return int(s)


def _terms(H: SparsePauliOp):
    """[(label, coeff)] sorted by support so commuting same-bond terms are adjacent."""
    n = H.num_qubits
    out = []
    for label, c in zip(H.paulis.to_labels(), H.coeffs):
        if abs(c.imag) > 1e-12:
            raise ValueError("Hamiltonian must be Hermitian (real Pauli coefficients)")
        supp = tuple(i for i in range(n) if label[n - 1 - i] != "I")
        out.append((supp, label, float(c.real)))
    out.sort(key=lambda x: (len(x[0]) == 0, x[0], x[1]))
    return [(label, c) for _, label, c in out]


def _sequence(H: SparsePauliOp, delta: float, steps: int):
    """[(label, theta)] for S2(delta)^steps, adjacent repeats merged (halves the gate count)."""
    terms = _terms(H)
    half = [(l, c * delta / 2) for l, c in terms]
    one = half + half[::-1]
    seq = []
    for _ in range(steps):
        for l, th in one:
            if seq and seq[-1][0] == l:
                seq[-1] = (l, seq[-1][1] + th)
            else:
                seq.append((l, th))
    return seq


def _controlled_pauli_rotation(qc: QuantumCircuit, sys, anc, label: str, theta: float):
    """Controlled e^{-i theta P}: basis change -> parity ladder -> CRZ(2 theta) -> uncompute."""
    n = len(sys)
    supp = [i for i in range(n) if label[n - 1 - i] != "I"]
    if not supp:                               # identity term: a relative phase on the ancilla
        qc.p(-theta, anc)
        return
    for i in supp:
        P = label[n - 1 - i]
        if P == "X":
            qc.h(sys[i])
        elif P == "Y":
            qc.sdg(sys[i])
            qc.h(sys[i])
    for a, b in zip(supp[:-1], supp[1:]):
        qc.cx(sys[a], sys[b])
    tgt = sys[supp[-1]]
    qc.rz(theta, tgt)
    qc.cx(anc, tgt)
    qc.rz(-theta, tgt)
    qc.cx(anc, tgt)
    for a, b in reversed(list(zip(supp[:-1], supp[1:]))):
        qc.cx(sys[a], sys[b])
    for i in supp:
        P = label[n - 1 - i]
        if P == "X":
            qc.h(sys[i])
        elif P == "Y":
            qc.h(sys[i])
            qc.s(sys[i])


def controlled_trotter(H: SparsePauliOp, t: float, evo: Trotter) -> QuantumCircuit:
    """Circuit on n system qubits + 1 ancilla (last qubit), same layout as the exact gate."""
    n = H.num_qubits
    qc = QuantumCircuit(n + 1, name=f"c-U_trot({t:.2f})")
    for label, th in _sequence(H, evo.delta, evo.steps(t)):
        _controlled_pauli_rotation(qc, list(range(n)), n, label, th)
    return qc


def trotter_step_unitary(H: SparsePauliOp, delta: float) -> np.ndarray:
    """The exact matrix of one S2(delta) step (reference for the quasi-energies)."""
    n = H.num_qubits
    U = np.eye(2**n, dtype=complex)
    for label, th in _sequence(H, delta, 1):
        P = SparsePauliOp(label).to_matrix()
        U = expm(-1j * th * P) @ U
    return U


def quasi_energies(H: SparsePauliOp, delta: float) -> np.ndarray:
    """Floquet quasi-energies eps with S2 = sum e^{-i eps delta} |v><v|, folded to (-pi/delta, pi/delta]."""
    lam = np.linalg.eigvals(trotter_step_unitary(H, delta))
    return np.sort(-np.angle(lam) / delta)


def trotter_spectrum(H: SparsePauliOp, psi: np.ndarray, delta: float, projectors: dict, tol: float = 1e-8):
    """Populated quasi-energy levels of |psi> under S2(delta), per symmetry sector.

    The reference a noiseless Trotterized experiment should reproduce exactly (vs. the exact
    spectrum, which differs by the Trotter error). Complex Schur form, not eig: S2 is unitary
    (normal), and Schur keeps eigenvectors orthonormal inside degenerate subspaces."""
    from scipy.linalg import schur
    S = trotter_step_unitary(H, delta)
    levels = []
    for m, P in projectors.items():
        vals, vecs = np.linalg.eigh(P.to_matrix())
        B = vecs[:, vals > 0.5]
        T, Z = schur(B.conj().T @ S @ B, output="complex")
        eps = -np.angle(np.diag(T)) / delta
        amps = np.abs((B @ Z).conj().T @ psi) ** 2
        merged = []
        for e, a in sorted(zip(eps, amps)):
            if merged and abs(merged[-1][0] - e) < 1e-6:
                merged[-1][1] += a
            else:
                merged.append([e, a])
        levels += [{"sector": m, "energy": float(e), "weight": float(a)} for e, a in merged if a > tol]
    return levels


def two_qubit_gate_count(H: SparsePauliOp, t: float, evo: Trotter) -> int:
    return controlled_trotter(H, t, evo).count_ops().get("cx", 0)
