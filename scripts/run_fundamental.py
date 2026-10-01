"""Fundamental task: validate the shadow-enhanced Hadamard test against exact references.

From ONE shared set of shots (ancilla in X / Y basis + random Pauli snapshots of the system):
  * complex Hadamard signal          g(t) = <psi|U|psi>
  * input-state energy               <H>_psi         (system marginal, H conserved)
  * conserved quantity               <M_z>_psi       (M_z = sum_i Z_i)
  * joint ancilla-system observables <psi|H U|psi>, <psi|M_z U|psi>
and show the error falls like 1/sqrt(shots).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shadow_hadamard.experiment import UNIFORM, estimate, run_experiment
from shadow_hadamard.model import default_problem

OUT = Path(__file__).resolve().parents[1] / "results"

QUANTITIES = [  # (label, getter on an estimate row, exact value function)
    ("Re g(t)", lambda r: r["signal"].value.real, lambda p, t: p.exact_signal(t).real),
    ("Im g(t)", lambda r: r["signal"].value.imag, lambda p, t: p.exact_signal(t).imag),
    ("<H>", lambda r: r["marg:H"].value, lambda p, t: p.exact_expectation("H")),
    ("<Mz>", lambda r: r["marg:Mz"].value, lambda p, t: p.exact_expectation("Mz")),
    ("Re <psi|H U|psi>", lambda r: r["joint:H"].value.real, lambda p, t: p.exact_joint("H", t).real),
    ("Im <psi|H U|psi>", lambda r: r["joint:H"].value.imag, lambda p, t: p.exact_joint("H", t).imag),
    ("Re <psi|Mz U|psi>", lambda r: r["joint:Mz"].value.real, lambda p, t: p.exact_joint("Mz", t).real),
    ("Im <psi|Mz U|psi>", lambda r: r["joint:Mz"].value.imag, lambda p, t: p.exact_joint("Mz", t).imag),
]
ERRS = [
    lambda r: r["signal"].stderr.real, lambda r: r["signal"].stderr.imag,
    lambda r: r["marg:H"].stderr, lambda r: r["marg:Mz"].stderr,
    lambda r: r["joint:H"].stderr.real, lambda r: r["joint:H"].stderr.imag,
    lambda r: r["joint:Mz"].stderr.real, lambda r: r["joint:Mz"].stderr.imag,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--shots", type=int, default=100_000, help="shots per ancilla basis for the table")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    prob, t = default_problem(), args.t
    names = ["H", "Mz"]

    # ---- 1. validation table -------------------------------------------------------------
    row = estimate(prob, run_experiment(prob, [t], args.shots, UNIFORM, seed=args.seed), UNIFORM, names)[0]
    table = []
    print(f"t = {t}, {2 * args.shots} total shots (X and Y ancilla bases), uniform Pauli shadows\n")
    print(f"{'quantity':20s} {'estimate':>10s} {'+/- 1 sigma':>11s} {'exact':>10s} {'z-score':>8s}")
    for (lab, get, ex), err in zip(QUANTITIES, ERRS):
        v, e, x = get(row), err(row), ex(prob, t)
        print(f"{lab:20s} {v:10.4f} {e:11.4f} {x:10.4f} {(v - x) / e:8.2f}")
        table.append({"quantity": lab, "estimate": v, "stderr": e, "exact": x})

    # ---- 2. convergence with shot count ---------------------------------------------------
    shot_list = np.unique(np.logspace(2, 5, 7).astype(int))
    rng = np.random.default_rng(args.seed)
    rmse = np.zeros((len(shot_list), len(QUANTITIES)))
    for i, N in enumerate(shot_list):
        errs = []
        for _ in range(args.reps):
            r = estimate(prob, run_experiment(prob, [t], int(N), UNIFORM, seed=int(rng.integers(2**31))),
                         UNIFORM, names)[0]
            errs.append([get(r) - ex(prob, t) for _, get, ex in QUANTITIES])
        rmse[i] = np.sqrt(np.mean(np.square(errs), axis=0))
        print(f"N = {2 * N:7d} total shots  RMSE: " + "  ".join(f"{x:.4f}" for x in rmse[i]))

    total = 2 * shot_list
    slopes = [np.polyfit(np.log(total), np.log(rmse[:, j]), 1)[0] for j in range(len(QUANTITIES))]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for j, (lab, _, _) in enumerate(QUANTITIES):
        ax.loglog(total, rmse[:, j], "o-", ms=4, label=f"{lab}  (slope {slopes[j]:.2f})")
    ax.loglog(total, 3 / np.sqrt(total), "k--", lw=1, label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("total shots N (shared by all quantities)")
    ax.set_ylabel(f"RMSE over {args.reps} repetitions")
    ax.set_title(f"Shadow-enhanced Hadamard test, 3-qubit XXZ, t = {t}")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fundamental_convergence.png", dpi=160)

    json.dump({"t": t, "shots_per_basis": args.shots, "table": table,
               "convergence": {"total_shots": total.tolist(), "rmse": rmse.tolist(),
                               "labels": [q[0] for q in QUANTITIES], "slopes": slopes}},
              open(OUT / "fundamental.json", "w"), indent=2)
    print("\nlog-log slopes:", np.round(slopes, 3), "(ideal -0.5)")
    print("saved", OUT / "fundamental_convergence.png")


if __name__ == "__main__":
    main()
