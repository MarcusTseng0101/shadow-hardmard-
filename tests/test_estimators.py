"""Estimators must be exactly unbiased: with infinite shots they reproduce the analytic references."""
import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from shadow_hadamard.experiment import PauliShadowScheme, estimate, run_experiment
from shadow_hadamard.model import default_problem

SCHEMES = [PauliShadowScheme(), PauliShadowScheme(0.2, 0.2, 0.6)]


@pytest.mark.parametrize("scheme", SCHEMES, ids=str)
def test_exact_estimators(scheme):
    prob = default_problem()
    prob.observables["X0"] = SparsePauliOp.from_sparse_list([("X", [0], 1.0)], 3)  # not conserved
    times = [0.0, 0.7, 2.3]
    rows = estimate(prob, run_experiment(prob, times, 0, scheme, mode="exact"), scheme)
    for r in rows:
        t = r["t"]
        assert np.isclose(r["signal"].value, prob.exact_signal(t), atol=1e-10)
        for nm in prob.observables:
            assert np.isclose(r[f"joint:{nm}"].value, prob.exact_joint(nm, t), atol=1e-10), nm
            assert np.isclose(r[f"marg:{nm}"].value, prob.exact_marginal(nm, t), atol=1e-10), nm
    # conserved quantities: the marginal is the input-state expectation at every time
    for nm in ["H", "Mz"]:
        for r in rows:
            assert np.isclose(r[f"marg:{nm}"].value, prob.exact_expectation(nm), atol=1e-10)


def test_z_only_scheme_flags_unmeasurable():
    prob = default_problem()
    scheme = PauliShadowScheme(0, 0, 1)
    rows = estimate(prob, run_experiment(prob, [0.5], 0, scheme, mode="exact"), scheme)
    assert np.isnan(rows[0]["marg:H"].value)            # XX, YY terms need X / Y bases
    assert np.isclose(rows[0]["joint:P[+1]"].value, prob.exact_joint("P[+1]", 0.5), atol=1e-10)


def test_sector_signals_sum_to_signal():
    prob = default_problem()
    t = 1.3
    total = sum(prob.exact_joint(f"P[{m:+d}]", t) for m in (3, 1, -1, -3))
    assert np.isclose(total, prob.exact_signal(t))


def test_error_bars_are_honest():
    """Across independent repetitions the errors must match the reported standard errors (z^2 ~ 1).
    Guards against correlated sampling between basis-setting circuits."""
    prob = default_problem()
    rng = np.random.default_rng(0)
    z2 = []
    for _ in range(40):
        r = estimate(prob, run_experiment(prob, [1.0], 3000, seed=int(rng.integers(2**31))))[0]
        g, e, x = r["signal"].value, r["signal"].stderr, prob.exact_signal(1.0)
        z2 += [((g.real - x.real) / e.real) ** 2, ((g.imag - x.imag) / e.imag) ** 2]
        m, me = r["joint:Mz"].value, r["joint:Mz"].stderr
        mx = prob.exact_joint("Mz", 1.0)
        z2 += [((m.real - mx.real) / me.real) ** 2]
    assert 0.6 < np.mean(z2) < 1.5, np.mean(z2)


def test_aer_run_is_statistically_consistent():
    prob = default_problem()
    rows = estimate(prob, run_experiment(prob, [1.0], 20000, seed=7))
    r = rows[0]
    g, e = r["signal"].value, r["signal"].stderr
    exact = prob.exact_signal(1.0)
    assert abs(g.real - exact.real) < 5 * e.real and abs(g.imag - exact.imag) < 5 * e.imag
