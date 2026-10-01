"""Hamiltonian, symmetry operators, input state and exact references."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def xxz_hamiltonian(n: int = 3, J: float = 1.0, delta: float = 0.5, h: float = 0.0,
                    periodic: bool = False) -> SparsePauliOp:
    """H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1}) + h sum_i Z_i."""
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        bonds.append((n - 1, 0))
    terms = []
    for i, j in bonds:
        terms += [("XX", [i, j], J), ("YY", [i, j], J), ("ZZ", [i, j], J * delta)]
    if h:
        terms += [("Z", [i], h) for i in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def total_z(n: int) -> SparsePauliOp:
    """M = sum_i Z_i  (= 2 S^z_total). Commutes with every XXZ Hamiltonian."""
    return SparsePauliOp.from_sparse_list([("Z", [i], 1.0) for i in range(n)], num_qubits=n)


def magnetization_projectors(n: int) -> dict[int, SparsePauliOp]:
    """Projectors P_m onto the sectors sum_i Z_i = m, written as sums of Z-strings."""
    idx = np.arange(2**n)
    m_of_idx = np.array([n - 2 * bin(k).count("1") for k in idx])  # bit 0 -> Z=+1
    projs = {}
    for m in sorted(set(m_of_idx), reverse=True):
        diag = (m_of_idx == m).astype(float)
        projs[int(m)] = SparsePauliOp.from_operator(np.diag(diag)).simplify()
    return projs


def product_state_circuit(thetas, phis) -> QuantumCircuit:
    """|psi> = prod_i Rz(phi_i) Ry(theta_i) |0>, an easy-to-prepare state that overlaps every sector."""
    n = len(thetas)
    qc = QuantumCircuit(n, name="prep")
    for i, (th, ph) in enumerate(zip(thetas, phis)):
        qc.ry(th, i)
        qc.rz(ph, i)
    return qc


@dataclass
class Problem:
    """Everything that defines one shadow-enhanced Hadamard test experiment."""
    H: SparsePauliOp
    prep: QuantumCircuit
    observables: dict[str, SparsePauliOp] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.H.num_qubits

    @property
    def psi(self) -> np.ndarray:
        return Statevector.from_instruction(self.prep).data

    def U(self, t: float) -> np.ndarray:
        E, V = np.linalg.eigh(self.H.to_matrix())
        return (V * np.exp(-1j * E * t)) @ V.conj().T

    # ---- exact references -------------------------------------------------
    def exact_signal(self, t: float) -> complex:
        """g(t) = <psi| e^{-iHt} |psi>."""
        psi = self.psi
        return complex(psi.conj() @ self.U(t) @ psi)

    def exact_joint(self, name: str, t: float) -> complex:
        """<psi| O e^{-iHt} |psi>  (ancilla-system correlator)."""
        psi, O = self.psi, self.observables[name].to_matrix()
        return complex(psi.conj() @ O @ self.U(t) @ psi)

    def exact_marginal(self, name: str, t: float) -> float:
        """(<O>_psi + <O>_{U psi}) / 2 : what the system register holds when the ancilla is ignored."""
        psi, O = self.psi, self.observables[name].to_matrix()
        phi = self.U(t) @ psi
        return float(np.real(psi.conj() @ O @ psi + phi.conj() @ O @ phi) / 2)

    def exact_expectation(self, name: str) -> float:
        psi, O = self.psi, self.observables[name].to_matrix()
        return float(np.real(psi.conj() @ O @ psi))

    def exact_spectrum(self, projectors: dict[int, SparsePauliOp], tol: float = 1e-8):
        """Populated levels of |psi>, resolved by symmetry sector.

        Returns a list of dicts {sector, energy, weight}; degenerate eigenvectors inside a
        sector are merged into one level (they are indistinguishable in the signal)."""
        psi = self.psi
        Hm = self.H.to_matrix()
        levels = []
        for m, P in projectors.items():
            Pm = P.to_matrix()
            # orthonormal basis of the sector, then diagonalise H inside it
            vals, vecs = np.linalg.eigh(Pm)
            B = vecs[:, vals > 0.5]
            E, W = np.linalg.eigh(B.conj().T @ Hm @ B)
            amps = np.abs((B @ W).conj().T @ psi) ** 2
            for e, w in _merge_degenerate(E, amps, tol=1e-6):
                if w > tol:
                    levels.append({"sector": m, "energy": e, "weight": w})
        return levels


def _merge_degenerate(E, w, tol):
    out = []
    for e, a in sorted(zip(E, w)):
        if out and abs(out[-1][0] - e) < tol:
            out[-1][1] += a
        else:
            out.append([e, a])
    return [(float(e), float(a)) for e, a in out]


def default_problem(J=1.0, delta=0.5, h=0.0, periodic=False,
                    thetas=(1.1, 2.0, 0.6), phis=(0.3, -0.8, 1.2)) -> Problem:
    """Three-qubit XXZ chain + a product input state. Parameters are free to change."""
    n = len(thetas)
    H = xxz_hamiltonian(n, J=J, delta=delta, h=h, periodic=periodic)
    obs = {"H": H, "Mz": total_z(n)}
    for m, P in magnetization_projectors(n).items():
        obs[f"P[{m:+d}]"] = P
    return Problem(H=H, prep=product_state_circuit(thetas, phis), observables=obs)
