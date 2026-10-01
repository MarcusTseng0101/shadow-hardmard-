"""Symmetry-resolved spectral analysis of shadow-Hadamard signals.

For a symmetry sector projector P_m that commutes with H,

    g_m(t) = <psi| P_m e^{-iHt} |psi> = sum_{k in sector m} |c_k|^2 e^{-i E_k t},

so every sector signal is a sum of at most dim(sector) damped-free exponentials with real,
non-negative amplitudes. g_m comes from the *same shots* as the ordinary Hadamard signal
g(t) = sum_m g_m(t): it is the ancilla-system correlator E[s * P_m_hat].

Pipeline per signal: Hermitian extension g(-t) = conj g(t) -> matrix pencil (ESPRIT-type) for
the frequencies -> non-negative least squares for the weights -> joint non-linear refinement ->
significance cut from the fit residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, nnls


@dataclass
class Peak:
    energy: float
    weight: float
    weight_err: float
    sector: int | None = None


def hermitian_extension(t: np.ndarray, y: np.ndarray):
    """Use g(-t) = conj g(t) (valid for Hermitian H, P_m) to double the record length for free."""
    assert np.isclose(t[0], 0.0), "grid must start at t = 0"
    return np.concatenate([-t[:0:-1], t]), np.concatenate([np.conj(y[:0:-1]), y])


def matrix_pencil(y: np.ndarray, order: int) -> np.ndarray:
    """Poles z_j of y_k ~ sum_j a_j z_j^k from the rank-`order` signal subspace of the Hankel matrix."""
    N = len(y)
    L = N // 2
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    _, _, Vh = np.linalg.svd(Y, full_matrices=False)
    W = Vh[:order].T
    return np.linalg.eigvals(np.linalg.pinv(W[:-1]) @ W[1:])


def _design(E, t, gamma: float = 0.0):
    return np.exp(-1j * np.outer(t, E)) * np.exp(-gamma * np.abs(t))[:, None]


def _nnls_weights(E, t, y, gamma: float = 0.0):
    A = _design(E, t, gamma)
    w, _ = nnls(np.vstack([A.real, A.imag]), np.concatenate([y.real, y.imag]))
    return w


def fit_signal(t, y, max_order: int, dt: float, refine: bool = True, n_sigma: float = 3.0,
               damped: bool = False):
    """Return significant peaks (energy, weight, weight_err) of a uniformly sampled signal.

    damped=False : unitary model, poles on the unit circle (noiseless data).
    damped=True  : y(t) = sum_j w_j e^{-i E_j t} e^{-gamma |t|} with one shared gamma >= 0.
                   Gate noise shrinks the signal roughly geometrically in circuit depth, i.e.
                   exponentially in t for a fixed Trotter step. Fitting gamma and reporting the
                   weights at t = 0 undoes that shrinkage (noise mitigation by extrapolation in
                   time). Energies are unaffected by a pure decay, so they need no correction.
    """
    t = np.asarray(t)
    te, ye = hermitian_extension(t, np.asarray(y))
    order = min(max_order, len(ye) // 3)
    gamma = 0.0
    if damped:
        # Poles from the one-sided record: the |t| kink makes the extended record non-exponential.
        z = matrix_pencil(np.asarray(y), min(max_order, len(y) // 3))
        gamma = float(np.median(np.clip(-np.log(np.abs(z)) / dt, 0, None)))
    else:
        z = matrix_pencil(ye, order)
    E = -np.angle(z) / dt                     # unitary evolution: poles lie on the unit circle
    w = _nnls_weights(E, te, ye, gamma)
    keep = w > 0
    E, w = E[keep], w[keep]
    if refine and len(E):
        k = len(E)

        def resid(p):
            g = p[2 * k] if damped else 0.0
            r = ye - _design(p[:k], te, g) @ p[k:2 * k]
            return np.concatenate([r.real, r.imag])
        p0 = np.concatenate([E, w] + ([[gamma]] if damped else []))
        lo = np.concatenate([E - np.pi / dt, np.zeros_like(w)] + ([[0.0]] if damped else []))
        hi = np.concatenate([E + np.pi / dt, np.full_like(w, 1.0)] + ([[np.inf]] if damped else []))
        sol = least_squares(resid, np.clip(p0, lo, np.where(np.isinf(hi), p0 + 1, hi)), bounds=(lo, hi))
        E, w = sol.x[:k], sol.x[k:2 * k]
        gamma = float(sol.x[2 * k]) if damped else 0.0
    # noise level from the residual -> standard error of a single weight (~ sigma / sqrt(N_points))
    r = ye - _design(E, te, gamma) @ w if len(E) else ye
    sigma = np.sqrt(np.mean(np.abs(r) ** 2) / 2)
    w_err = sigma / np.sqrt(len(te))
    peaks = [Peak(float(e), float(a), float(w_err)) for e, a in zip(E, w) if a > max(n_sigma * w_err, 1e-6)]
    return sorted(peaks, key=lambda p: p.energy)


def symmetry_resolved_spectrum(rows: list[dict], sectors: dict[int, int], **kw) -> list[Peak]:
    """Fit every sector signal g_m(t) separately. `sectors` maps label m -> sector dimension,
    which bounds the number of levels (model order) in that sector."""
    t = np.array([r["t"] for r in rows])
    dt = t[1] - t[0]
    peaks = []
    for m, dim in sectors.items():
        y = np.array([r[f"joint:P[{m:+d}]"].value for r in rows])
        for p in fit_signal(t, y, max_order=dim, dt=dt, **kw):
            p.sector = m
            peaks.append(p)
    return peaks


def ancilla_only_spectrum(rows: list[dict], max_order: int, **kw) -> list[Peak]:
    """Baseline: the ordinary Hadamard test signal g(t) alone (no labels, degenerate levels merge)."""
    t = np.array([r["t"] for r in rows])
    y = np.array([r["signal"].value for r in rows])
    return fit_signal(t, y, max_order=max_order, dt=t[1] - t[0], **kw)


def match_levels(peaks: list[Peak], exact: list[dict], by_sector: bool = True):
    """Pair each exact populated level with the closest reconstructed peak (same sector if labelled).

    Returns rows (exact_level, peak or None); extra peaks are returned separately as spurious."""
    used, pairs = set(), []
    for lev in sorted(exact, key=lambda l: -l["weight"]):
        cands = [(abs(p.energy - lev["energy"]), i) for i, p in enumerate(peaks)
                 if i not in used and (not by_sector or p.sector == lev["sector"])]
        cands = [c for c in cands if c[0] < 0.5]
        if cands:
            _, i = min(cands)
            used.add(i)
            pairs.append((lev, peaks[i]))
        else:
            pairs.append((lev, None))
    spurious = [p for i, p in enumerate(peaks) if i not in used]
    return pairs, spurious


def merge_exact_by_energy(exact: list[dict], tol=1e-6):
    """What the ancilla-only signal can see: levels with equal energy collapse into one peak."""
    out = []
    for lev in sorted(exact, key=lambda l: l["energy"]):
        if out and abs(out[-1]["energy"] - lev["energy"]) < tol:
            out[-1]["weight"] += lev["weight"]
        else:
            out.append({"sector": None, "energy": lev["energy"], "weight": lev["weight"]})
    return out
