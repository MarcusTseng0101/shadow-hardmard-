"""Advanced task: symmetry-resolved spectral analyzer for a three-qubit XXZ chain.

Under a fixed total shot budget, reconstruct the populated energy levels, their spectral
weights |<E_k|psi>|^2, and the S^z symmetry label of every peak - using only the recycled
system register of the Hadamard test (sector signals <psi|P_m U(t)|psi>).

Compares three system-measurement strategies at equal budget and, as a baseline, the plain
Hadamard test (ancilla only), which cannot label peaks and merges symmetry-degenerate levels.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shadow_hadamard.experiment import PauliShadowScheme, estimate, pooled_marginal, run_experiment
from shadow_hadamard.model import default_problem, magnetization_projectors
from shadow_hadamard.spectral import (ancilla_only_spectrum, match_levels, merge_exact_by_energy,
                                      symmetry_resolved_spectrum)

OUT = Path(__file__).resolve().parents[1] / "results"
SCHEMES = {
    "uniform shadows": PauliShadowScheme(1 / 3, 1 / 3, 1 / 3),
    "Z-biased shadows": PauliShadowScheme(0.15, 0.15, 0.70),
    "Z-basis only": PauliShadowScheme(0.0, 0.0, 1.0),
}
SECTOR_COLORS = {3: "#1b9e77", 1: "#d95f02", -1: "#7570b3", -3: "#e7298a"}
E_TOL = 0.15  # a level counts as resolved if a peak with the right label lies within this energy


def analyze(prob, sectors, exact, times, budget, scheme, seed):
    shots = budget // (2 * len(times))
    rows = estimate(prob, run_experiment(prob, times, shots, scheme, seed=seed), scheme)
    peaks = symmetry_resolved_spectrum(rows, sectors)
    pairs, spurious = match_levels(peaks, exact)
    base = ancilla_only_spectrum(rows, max_order=2**prob.n)
    return rows, peaks, pairs, spurious, base


def metrics(prob, pairs, spurious, peaks, rows):
    found = [(l, p) for l, p in pairs if p is not None and abs(p.energy - l["energy"]) < E_TOL]
    H_shadow = pooled_marginal(rows, "H").value
    return {
        "resolved_fraction": len(found) / len(pairs),
        "resolved_weight_fraction": sum(l["weight"] for l, _ in found),
        "mean_abs_energy_err": float(np.mean([abs(p.energy - l["energy"]) for l, p in found])) if found else np.nan,
        "mean_abs_weight_err": float(np.mean([abs(p.weight - l["weight"]) for l, p in found])) if found else np.nan,
        "n_spurious": len(spurious),
        "energy_from_spectrum_err": abs(sum(p.energy * p.weight for p in peaks) - prob.exact_expectation("H")),
        "energy_from_shadow_err": abs(H_shadow - prob.exact_expectation("H")) if np.isfinite(H_shadow) else np.nan,
    }


def plot_showcase(prob, exact, rows, peaks, base, scheme_name, budget, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    for m, col in SECTOR_COLORS.items():
        lv = [l for l in exact if l["sector"] == m]
        pk = [p for p in peaks if p.sector == m]
        off = {3: -0.09, 1: -0.03, -1: 0.03, -3: 0.09}[m]
        ax.vlines([l["energy"] + off for l in lv], 0, [l["weight"] for l in lv], color=col, lw=6, alpha=0.25)
        ax.errorbar([p.energy + off for p in pk], [p.weight for p in pk], yerr=[p.weight_err for p in pk],
                    fmt="o", color=col, ms=6, capsize=3, label=f"$S^z = {m/2:+.1f}$")
    bexact = merge_exact_by_energy(exact)
    ax.plot([l["energy"] for l in bexact], [l["weight"] for l in bexact], "k_", ms=22, mew=1.5,
            label="ancilla-only: exact (merged)")
    ax.plot([p.energy for p in base], [p.weight for p in base], "kx", ms=8, label="ancilla-only: estimate")
    ax.set_xlabel("energy $E$")
    ax.set_ylabel(r"spectral weight $|\langle E|\psi\rangle|^2$")
    ax.set_title(f"Symmetry-resolved spectrum ({scheme_name}, {budget:.0e} shots)\n"
                 "bars = exact per sector, dots = reconstructed (offset per sector for visibility)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.grid(alpha=0.3)

    ax = axs[1]
    t = np.array([r["t"] for r in rows])
    tf = np.linspace(0, t[-1], 400)
    for m, col in SECTOR_COLORS.items():
        v = np.array([r[f"joint:P[{m:+d}]"].value.real for r in rows])
        e = np.array([r[f"joint:P[{m:+d}]"].stderr.real for r in rows])
        ax.errorbar(t, v, yerr=e, fmt=".", color=col, ms=4, lw=0.8)
        ax.plot(tf, [prob.exact_joint(f"P[{m:+d}]", x).real for x in tf], color=col, lw=1,
                label=rf"Re$\langle\psi|P_{{{m:+d}}}U(t)|\psi\rangle$")
    ax.set_xlabel("evolution time $t$")
    ax.set_title("Sector signals from the recycled register (points) vs exact (lines)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--h", type=float, default=0.0, help="longitudinal field (lifts +/-S^z degeneracy)")
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--K", type=int, default=40, help="number of evolution times")
    ap.add_argument("--budget", type=int, default=200_000, help="total shots for the showcase run")
    ap.add_argument("--budgets", type=int, nargs="*", default=[50_000, 200_000, 1_000_000])
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _h03")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    prob = default_problem(J=args.J, delta=args.delta, h=args.h, periodic=args.periodic)
    projs = magnetization_projectors(prob.n)
    sectors = {m: int(round(np.trace(P.to_matrix()).real)) for m, P in projs.items()}
    exact = prob.exact_spectrum(projs)
    # Nyquist: |E| <= ||H|| <= sum |coeffs|, so dt < pi / that bound avoids aliasing.
    dt = 0.9 * np.pi / float(np.sum(np.abs(prob.H.coeffs)))
    times = dt * np.arange(args.K)
    print(f"XXZ: J={args.J} delta={args.delta} h={args.h} periodic={args.periodic};  dt={dt:.3f}, "
          f"K={args.K}, T_max={times[-1]:.1f}")
    print("exact populated levels:")
    for l in exact:
        print(f"  S^z={l['sector'] / 2:+.1f}  E={l['energy']:+.4f}  w={l['weight']:.4f}")

    # ---- showcase run -----------------------------------------------------------------------
    report = {"model": vars(args), "dt": dt, "exact_levels": exact, "showcase": {}, "sweep": {}}
    rng = np.random.default_rng(args.seed)
    for name, scheme in SCHEMES.items():
        rows, peaks, pairs, spurious, base = analyze(prob, sectors, exact, times, args.budget, scheme,
                                                     int(rng.integers(2**31)))
        met = metrics(prob, pairs, spurious, peaks, rows)
        print(f"\n== {name}: {scheme}  budget {args.budget}")
        print(f"  {'S^z':>5s} {'E exact':>8s} {'w exact':>8s} | {'E est':>8s} {'w est':>16s}")
        for l, p in sorted(pairs, key=lambda x: (-x[0]['sector'], x[0]['energy'])):
            est = "(not resolved)" if p is None else f"{p.energy:+8.4f} {p.weight:8.4f} +- {p.weight_err:.4f}"
            print(f"  {l['sector'] / 2:+5.1f} {l['energy']:+8.4f} {l['weight']:8.4f} | {est}")
        if spurious:
            print("  spurious:", [(p.sector / 2, round(p.energy, 3), round(p.weight, 4)) for p in spurious])
        print("  ancilla-only peaks (no labels):", [(round(p.energy, 3), round(p.weight, 3)) for p in base])
        print("  metrics:", {k: round(v, 4) for k, v in met.items()})
        slug = name.split()[0].lower().replace("-", "")
        plot_showcase(prob, exact, rows, peaks, base, name, args.budget, OUT / f"advanced_spectrum_{slug}{args.tag}.png")
        report["showcase"][name] = {
            "metrics": met,
            "peaks": [vars(p) for p in peaks],
            "ancilla_only_peaks": [vars(p) for p in base],
        }

    # ---- budget sweep, repeated -------------------------------------------------------------
    keys = ["resolved_fraction", "mean_abs_energy_err", "mean_abs_weight_err", "n_spurious"]
    for name, scheme in SCHEMES.items():
        report["sweep"][name] = {}
        for B in args.budgets:
            ms = []
            for _ in range(args.reps):
                rows, peaks, pairs, spurious, _ = analyze(prob, sectors, exact, times, B, scheme,
                                                          int(rng.integers(2**31)))
                ms.append(metrics(prob, pairs, spurious, peaks, rows))
            # energy_from_shadow_err is NaN for Z-only (XX/YY unmeasurable): keep it NaN, no warning
            agg = {k: float(np.mean(v)) if np.all(np.isnan(v := [m[k] for m in ms])) else float(np.nanmean(v))
                   for k in ms[0]}
            report["sweep"][name][B] = agg
            print(f"sweep {name:17s} B={B:>8d}: " + "  ".join(f"{k}={agg[k]:.4f}" for k in keys))

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for name in SCHEMES:
        Bs = list(report["sweep"][name])
        for ax, k, lab in zip(axs, ["resolved_fraction", "mean_abs_energy_err", "mean_abs_weight_err"],
                              ["fraction of 8 labelled levels resolved", "mean |E error|", "mean |weight error|"]):
            ax.plot(Bs, [report["sweep"][name][b][k] for b in Bs], "o-", label=name)
            ax.set_xscale("log")
            ax.set_xlabel("total shot budget")
            ax.set_title(lab, fontsize=10)
            ax.grid(alpha=0.3, which="both")
    axs[1].set_yscale("log")
    axs[2].set_yscale("log")
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Measurement strategy comparison (mean of {args.reps} runs, K={args.K} times)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"advanced_strategies{args.tag}.png", dpi=160)
    json.dump(report, open(OUT / f"advanced{args.tag}.json", "w"), indent=2, default=float)
    print("saved results to", OUT)


if __name__ == "__main__":
    main()
