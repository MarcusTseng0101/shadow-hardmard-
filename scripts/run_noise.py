"""Gate-level (Trotter) evolution, noise, and noise mitigation.

  1. Trotter error vs step size, and the CX cost that buys it (hardware resource estimate).
  2. Noiseless Trotterized spectroscopy: reproduces the Trotter quasi-energies exactly; the
     remaining gap to the true energies is pure Trotter error.
  3. Noise sweep (depolarizing gates + readout error) x three analyses:
       raw | readout-error mitigation (REM) | REM + damped spectral fit (extrapolate weights to t=0)
  4. The fundamental quantities at t = 1 under noise, raw vs REM.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shadow_hadamard.experiment import UNIFORM, PauliShadowScheme, estimate, pooled_marginal, run_experiment
from shadow_hadamard.model import default_problem, magnetization_projectors
from shadow_hadamard.noise import NoiseLevel, calibrate_readout, mitigate
from shadow_hadamard.spectral import match_levels, symmetry_resolved_spectrum
from shadow_hadamard.trotter import (Trotter, controlled_trotter, quasi_energies, trotter_spectrum,
                                     trotter_step_unitary)

OUT = Path(__file__).resolve().parents[1] / "results"
LEVELS = [NoiseLevel("noiseless", 0, 0, 0),
          NoiseLevel("p2=1e-4", 1e-5, 1e-4, 0.005),
          NoiseLevel("p2=1e-3", 1e-4, 1e-3, 0.01),
          NoiseLevel("p2=3e-3", 3e-4, 3e-3, 0.02)]      # GUESS: a rough "current hardware" point
METHODS = ["raw", "REM", "REM + damped fit"]
Z_ONLY = PauliShadowScheme(0, 0, 1)
E_TOL = 0.15


def metrics(peaks, ref):
    pairs, spurious = match_levels(peaks, ref)
    found = [(l, p) for l, p in pairs if p is not None and abs(p.energy - l["energy"]) < E_TOL]
    return {"resolved": len(found) / len(pairs),
            "dE": float(np.mean([abs(p.energy - l["energy"]) for l, p in found])) if found else np.nan,
            "dw": float(np.mean([abs(p.weight - l["weight"]) for l, p in found])) if found else np.nan,
            "sum_w": float(sum(p.weight for p in peaks)),
            "spurious": len(spurious)}


def analyses(prob, data, nm, sectors):
    out = {"raw": (estimate(prob, data, Z_ONLY), False)}
    if nm is not None:
        m = estimate(prob, mitigate(data, calibrate_readout(nm, prob.n + 1)), Z_ONLY)
        out["REM"] = (m, False)
        out["REM + damped fit"] = (m, True)
    else:
        out["REM"] = out["REM + damped fit"] = out["raw"]
    return {k: (rows, symmetry_resolved_spectrum(rows, sectors, damped=d)) for k, (rows, d) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--steps-per-sample", type=int, default=2)
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prob = default_problem()
    projs = magnetization_projectors(prob.n)
    sectors = {m: int(round(np.trace(P.to_matrix()).real)) for m, P in projs.items()}
    exact = prob.exact_spectrum(projs)
    E_exact = np.sort(np.linalg.eigvalsh(prob.H.to_matrix()))
    dt = 0.9 * np.pi / float(np.sum(np.abs(prob.H.coeffs)))
    delta = dt / args.steps_per_sample
    evo = Trotter(delta)
    times = dt * np.arange(args.K)
    ref = trotter_spectrum(prob.H, prob.psi, delta, projs)
    report = {"dt": dt, "delta": delta, "K": args.K, "budget": args.budget, "reps": args.reps}

    # ---- 1. Trotter error and cost ------------------------------------------------------------
    print("== 1. Trotter error vs step (second order) and CX cost")
    deltas = dt / np.array([1, 2, 3, 4, 6, 8])
    err = [float(np.max(np.abs(quasi_energies(prob.H, d) - E_exact))) for d in deltas]
    cx = [controlled_trotter(prob.H, times[-1], Trotter(d)).count_ops().get("cx", 0) for d in deltas]
    slope = np.polyfit(np.log(deltas), np.log(err), 1)[0]
    for d, e, c in zip(deltas, err, cx):
        print(f"  delta={d:.3f}  max|eps-E|={e:.4f}  CX at t_max={c}")
    print(f"  log-log slope {slope:.2f} (second order -> 2)")
    report["trotter"] = {"delta": deltas.tolist(), "max_err": err, "cx_at_tmax": cx, "slope": slope}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].loglog(deltas, err, "o-")
    ax[0].loglog(deltas, err[0] * (deltas / deltas[0]) ** 2, "k--", lw=1, label=r"$\propto\delta^2$")
    ax[0].set_xlabel(r"Trotter step $\delta$")
    ax[0].set_ylabel(r"max $|\varepsilon_k - E_k|$")
    ax[0].set_title(f"Quasi-energy error (slope {slope:.2f})", fontsize=10)
    ax[0].legend(frameon=False)
    from matplotlib.ticker import NullFormatter
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    cx_t = [controlled_trotter(prob.H, t, evo).count_ops().get("cx", 0) for t in times]
    ax[1].plot(times, cx_t, ".-")
    ax[1].set_xlabel("evolution time $t$")
    ax[1].set_ylabel("CX gates in controlled-U")
    ax[1].set_title(rf"Circuit cost, $\delta$={delta:.3f} (hardware resource estimate)", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "trotter_error.png", dpi=160)
    plt.close(fig)
    report["cx_per_time"] = dict(zip([round(t, 3) for t in times], cx_t))

    # ---- 2+3. noise sweep ---------------------------------------------------------------------
    print(f"\n== 2/3. Noise sweep: Trotter delta={delta:.3f}, K={args.K}, Z-basis system, "
          f"{args.budget} shots, {args.reps} reps; reference = Trotter quasi-energy spectrum")
    sweep, showcase = {}, None
    for lev in LEVELS:
        nm = lev.model() if lev.p2 or lev.p_ro else None
        res = {m: [] for m in METHODS}
        res_exact = {m: [] for m in METHODS}
        for r in range(args.reps):
            data = run_experiment(prob, times, args.budget // (2 * args.K), Z_ONLY,
                                  seed=int(rng.integers(2**31)), evolution=evo, noise_model=nm)
            an = analyses(prob, data, nm, sectors)
            for m in METHODS:
                res[m].append(metrics(an[m][1], ref))
                res_exact[m].append(metrics(an[m][1], exact))
            if lev.name == "p2=1e-3" and r == 0:
                showcase = an
        sweep[lev.name] = {m: {k: float(np.nanmean([x[k] for x in res[m]])) for k in res[m][0]} for m in METHODS}
        sweep[lev.name]["vs_exact_energies"] = {
            m: float(np.nanmean([x["dE"] for x in res_exact[m]])) for m in METHODS}
        for m in METHODS:
            s = sweep[lev.name][m]
            print(f"  {lev.name:10s} {m:17s} resolved {s['resolved']:.2f}  |dE| {s['dE']:.4f}  "
                  f"|dw| {s['dw']:.4f}  sum w {s['sum_w']:.3f}  spurious {s['spurious']:.1f}")
    report["sweep"] = sweep

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(LEVELS))
    for i, m in enumerate(METHODS):
        for ax, k, lab in zip(axs, ["dw", "sum_w", "dE"],
                              ["mean |weight error|", r"total weight $\sum w$ (exact: 1)", "mean |E error|"]):
            ax.plot(x + (i - 1) * 0.08, [sweep[l.name][m][k] for l in LEVELS], "o-", label=m)
            ax.set_xticks(x, [l.name for l in LEVELS], fontsize=8)
            ax.set_title(lab, fontsize=10)
            ax.grid(alpha=0.3)
    axs[0].set_yscale("log")
    axs[2].set_yscale("log")
    axs[1].axhline(1, color="k", lw=0.8, ls="--")
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Noise mitigation for symmetry-resolved spectroscopy (Trotter, mean of {args.reps} runs)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "noise_mitigation.png", dpi=160)
    plt.close(fig)

    if showcase:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        S = trotter_step_unitary(prob.H, delta)
        P = projs[1].to_matrix()
        psi = prob.psi
        tt = np.arange(args.K * args.steps_per_sample) * delta
        ex = [psi.conj() @ P @ np.linalg.matrix_power(S, k) @ psi for k in range(len(tt))]
        ax.plot(tt, np.real(ex), "k-", lw=1, label="noiseless Trotter (exact)")
        for m, mk in [("raw", "o"), ("REM", "s")]:
            rows = showcase[m][0]
            ax.plot(times, [r["joint:P[+1]"].value.real for r in rows], mk, ms=4, label=f"p2=1e-3, {m}")
        ax.set_xlabel("evolution time $t$")
        ax.set_ylabel(r"Re$\langle\psi|P_{+1}U(t)|\psi\rangle$")
        ax.set_title("Gate noise damps the signal with depth; readout mitigation cannot undo it", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "noise_signal.png", dpi=160)
        plt.close(fig)

    # ---- 4. fundamental quantities under noise ------------------------------------------------
    t, d4 = 1.0, Trotter(0.25)
    S4 = np.linalg.matrix_power(trotter_step_unitary(prob.H, 0.25), 4)
    psi = prob.psi
    refs = {"Re g": (psi.conj() @ S4 @ psi).real, "Im g": (psi.conj() @ S4 @ psi).imag,
            "<Mz>": prob.exact_expectation("Mz"),
            "Re <Mz U>": (psi.conj() @ prob.observables["Mz"].to_matrix() @ S4 @ psi).real}
    print(f"\n== 4. Fundamental quantities at t={t} under p2=1e-3 noise (uniform shadows, Trotter delta=0.25,"
          f" {controlled_trotter(prob.H, t, d4).count_ops().get('cx', 0)} CX)")
    nm = LEVELS[2].model()
    data = run_experiment(prob, [t], 100_000, UNIFORM, seed=int(rng.integers(2**31)), evolution=d4, noise_model=nm)
    rows = {"raw": estimate(prob, data, UNIFORM)[0],
            "REM": estimate(prob, mitigate(data, calibrate_readout(nm, prob.n + 1)), UNIFORM)[0]}
    get = {"Re g": lambda r: (r["signal"].value.real, r["signal"].stderr.real),
           "Im g": lambda r: (r["signal"].value.imag, r["signal"].stderr.imag),
           "<Mz>": lambda r: (r["marg:Mz"].value, r["marg:Mz"].stderr),
           "Re <Mz U>": lambda r: (r["joint:Mz"].value.real, r["joint:Mz"].stderr.real)}
    fund = {}
    print(f"  {'quantity':10s} {'reference':>9s} {'raw':>16s} {'REM':>16s}")
    for q, ref_v in refs.items():
        (vr, er), (vm, em) = get[q](rows["raw"]), get[q](rows["REM"])
        print(f"  {q:10s} {ref_v:9.4f} {vr:9.4f}+-{er:.4f} {vm:9.4f}+-{em:.4f}")
        fund[q] = {"reference": float(ref_v), "raw": [vr, er], "REM": [vm, em]}
    report["fundamental_noisy"] = fund

    json.dump(report, open(OUT / "noise.json", "w"), indent=2, default=float)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
