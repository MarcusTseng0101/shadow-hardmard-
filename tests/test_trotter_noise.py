"""Gate-level evolution, readout mitigation and the damped spectral fit."""
import numpy as np
import pytest
from qiskit.quantum_info import Operator

from shadow_hadamard.experiment import UNIFORM, TimeData, estimate, run_experiment
from shadow_hadamard.model import default_problem, total_z, xxz_hamiltonian
from shadow_hadamard.noise import NoiseLevel, calibrate_readout, mitigate
from shadow_hadamard.spectral import fit_signal
from shadow_hadamard.trotter import (Trotter, controlled_trotter, quasi_energies,
                                     trotter_step_unitary)


@pytest.mark.parametrize("h", [0.0, 0.3])
def test_circuit_is_exactly_the_controlled_trotter_unitary(h):
    H = xxz_hamiltonian(3, J=1.0, delta=0.5, h=h)
    evo = Trotter(0.25)
    S = trotter_step_unitary(H, 0.25)
    for steps in (0, 1, 3):
        U = np.linalg.matrix_power(S, steps)
        D = U.shape[0]
        CU = np.block([[np.eye(D), np.zeros((D, D))], [np.zeros((D, D)), U]])
        got = Operator(controlled_trotter(H, steps * 0.25, evo)).data
        assert np.allclose(got, CU, atol=1e-10)          # including the relative phase


def test_trotter_step_conserves_magnetization():
    H = xxz_hamiltonian(3, J=1.0, delta=0.5, h=0.2)
    S, M = trotter_step_unitary(H, 0.5), total_z(3).to_matrix()
    assert np.allclose(S @ M, M @ S, atol=1e-12)          # symmetry labels survive Trotterization


def test_quasi_energies_converge_quadratically():
    H = xxz_hamiltonian(3)
    exact = np.sort(np.linalg.eigvalsh(H.to_matrix()))
    errs = [np.max(np.abs(quasi_energies(H, d) - exact)) for d in (0.2, 0.1)]
    assert errs[1] < errs[0] / 3.0                       # second order: ~4x per halving


def test_estimators_unbiased_on_trotter_circuit():
    prob = default_problem()
    evo = Trotter(0.25)
    t = 1.0
    S = np.linalg.matrix_power(trotter_step_unitary(prob.H, 0.25), 4)
    psi = prob.psi
    row = estimate(prob, run_experiment(prob, [t], 0, UNIFORM, mode="exact", evolution=evo))[0]
    assert np.isclose(row["signal"].value, psi.conj() @ S @ psi, atol=1e-10)
    P = prob.observables["P[+1]"].to_matrix()
    assert np.isclose(row["joint:P[+1]"].value, psi.conj() @ P @ S @ psi, atol=1e-10)


def test_mitigate_inverts_a_known_confusion():
    rng = np.random.default_rng(0)
    p = rng.random((2, 8))
    p /= p.sum()
    A = [np.array([[0.97, 0.05], [0.03, 0.95]]), np.array([[0.9, 0.1], [0.1, 0.9]]),
         np.array([[0.99, 0.02], [0.01, 0.98]]), np.array([[0.95, 0.04], [0.05, 0.96]])]
    noisy = p.reshape([2] * 4)
    for q, Aq in enumerate(A):                           # apply the confusion, qubit q = axis 3 - q
        noisy = np.moveaxis(np.tensordot(Aq, noisy, axes=([1], [3 - q])), 0, 3 - q)
    td = TimeData(t=0.0, counts={("X", ("Z",) * 3): noisy.reshape(2, 8)})
    back = mitigate([td], A)[0].counts[("X", ("Z",) * 3)]
    assert np.allclose(back, p, atol=1e-12)


def test_readout_calibration_recovers_the_flip_rate():
    A = calibrate_readout(NoiseLevel("ro", 0, 0, 0.03).model(), 4, shots=40000)
    for Aq in A:
        assert abs(Aq[1, 0] - 0.03) < 0.005 and abs(Aq[0, 1] - 0.03) < 0.005


def test_damped_fit_recovers_undamped_weights():
    dt = 0.3
    t = dt * np.arange(40)
    E, w, gamma = np.array([-1.2, 0.7]), np.array([0.6, 0.3]), 0.05
    y = (w * np.exp(-1j * np.outer(t, E))).sum(1) * np.exp(-gamma * t)
    plain = fit_signal(t, y, 2, dt)
    damped = fit_signal(t, y, 2, dt, damped=True)
    assert np.allclose(sorted(p.energy for p in damped), sorted(E), atol=1e-4)
    assert np.allclose(sorted(p.weight for p in damped), sorted(w), atol=1e-3)
    assert sum(p.weight for p in plain) < 0.85 * w.sum()  # the unitary fit under-reports weight
