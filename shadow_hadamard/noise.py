"""Noise model and readout-error mitigation for the shadow-Hadamard experiment.

Noise model (all qubits identical, a deliberately simple stand-in for hardware):
    1-qubit gates : depolarizing p1
    CX            : depolarizing p2
    measurement   : symmetric bit flip p_ro

Readout mitigation (tensored): calibrate each qubit's 2x2 confusion matrix A_q from an all-|0>
and an all-|1> circuit, then apply (A_{n} x ... x A_0)^{-1} to every outcome histogram before the
estimators see it. The estimators are linear in the histogram, so the mitigated estimate is
unbiased for readout error; the result is a quasi-probability (entries may be slightly negative).
Gate noise is NOT touched here - see spectral.fit_signal(damped=True) for that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from .experiment import TimeData

ONE_QUBIT_GATES = ["h", "s", "sdg", "rz", "ry", "x", "p"]


@dataclass(frozen=True)
class NoiseLevel:
    name: str
    p1: float
    p2: float
    p_ro: float

    def model(self):
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        nm = NoiseModel()
        if self.p1:
            nm.add_all_qubit_quantum_error(depolarizing_error(self.p1, 1), ONE_QUBIT_GATES)
        if self.p2:
            nm.add_all_qubit_quantum_error(depolarizing_error(self.p2, 2), ["cx"])
        if self.p_ro:
            r = self.p_ro
            nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
        return nm


def calibrate_readout(noise_model, n_qubits: int, shots: int = 20000, seed: int = 0) -> list[np.ndarray]:
    """Per-qubit confusion matrices A_q[read, prepared] from all-0 and all-1 preparations."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    flips = np.zeros((n_qubits, 2))                    # P(read != prepared) for prepared 0 / 1
    for prep in (0, 1):
        qc = QuantumCircuit(n_qubits, n_qubits)
        if prep:
            qc.x(range(n_qubits))
        qc.measure(range(n_qubits), range(n_qubits))
        counts = sim.run(qc, shots=shots, seed_simulator=seed + prep).result().get_counts()
        for key, c in counts.items():
            v = int(key, 2)
            for q in range(n_qubits):
                if ((v >> q) & 1) != prep:
                    flips[q, prep] += c / shots
    return [np.array([[1 - f0, f1], [f0, 1 - f1]]) for f0, f1 in flips]


def mitigate(data: list[TimeData], confusion: list[np.ndarray]) -> list[TimeData]:
    """Apply the tensored inverse confusion matrix to every (2, 2^n) histogram.

    Histogram index v = anc * 2^n + sys, bit q of v is qubit q (ancilla = qubit n). Reshaped to
    [2]*(n+1) in C order, axis k holds bit (n - k), i.e. qubit n - k."""
    inv = [np.linalg.inv(A) for A in confusion]
    nq = len(inv)
    out = []
    for td in data:
        new = {}
        for key, C in td.counts.items():
            p = C.reshape([2] * nq)
            for q in range(nq):
                axis = nq - 1 - q
                p = np.moveaxis(np.tensordot(inv[q], p, axes=([1], [axis])), 0, axis)
            new[key] = p.reshape(C.shape)
        out.append(TimeData(t=td.t, counts=new))
    return out
