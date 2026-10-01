"""Shadow-enhanced Hadamard test on IBM Quantum hardware (or a fake backend for a dry run).

    python -m scripts.run_hardware --fake            # local dry run on a fake backend with real noise data
    python -m scripts.run_hardware                   # real device: uses the account saved with
                                                     # QiskitRuntimeService.save_account(...)

Only SHORT times are feasible on hardware (see results/trotter_error.png): t = 0.5, 1.0 with a
Trotter step of 0.25 is 2-4 steps, 80-160 CX before routing.

Layout discipline: the body (prep + H + controlled-U + ancilla rotation) is transpiled ONCE per
(time, ancilla basis); the random system-basis rotations and measurements are then appended in
native gates (rz, sx) on the body's FINAL physical qubits. Every random setting therefore reads
out the same physical qubits, so a single readout calibration on those qubits is valid for all.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager

from shadow_hadamard.experiment import UNIFORM, TimeData, estimate
from shadow_hadamard.model import default_problem
from shadow_hadamard.noise import mitigate
from shadow_hadamard.trotter import Trotter, controlled_trotter, trotter_step_unitary

OUT = Path(__file__).resolve().parents[1] / "results"
PI = np.pi


def body_circuit(prob, t, anc_basis, evo):
    n = prob.n
    q = QuantumRegister(n + 1, "q")
    qc = QuantumCircuit(q)
    qc.compose(prob.prep, qubits=list(range(n)), inplace=True)
    qc.h(n)
    qc.compose(controlled_trotter(prob.H, t, evo), qubits=list(range(n + 1)), inplace=True)
    if anc_basis == "Y":
        qc.sdg(n)
    qc.h(n)
    return qc


def _native_rotate(qc, basis, phys):
    """X: H = RZ(pi/2) SX RZ(pi/2); Y: S^dag then H, S^dag = RZ(-pi/2) (global phases dropped)."""
    if basis == "Z":
        return
    if basis == "Y":
        qc.rz(-PI / 2, phys)
    qc.rz(PI / 2, phys)
    qc.sx(phys)
    qc.rz(PI / 2, phys)


def measured(tbody, phys, setting, n):
    qc = tbody.copy()
    c = ClassicalRegister(n + 1, "c")
    qc.add_register(c)
    for i, b in enumerate(setting):
        _native_rotate(qc, b, phys[i])
    qc.measure([phys[i] for i in range(n + 1)], c)
    return qc


def calibration(width, phys, n, prep):
    qc = QuantumCircuit(QuantumRegister(width, "q"), ClassicalRegister(n + 1, "c"))
    if prep:
        for p in phys:
            qc.x(p)
    qc.measure(list(phys), qc.cregs[0])
    return qc


def to_hist(counts, n):
    arr = np.zeros((2, 2**n))
    for key, c in counts.items():
        v = int(key, 2)
        arr[(v >> n) & 1, v & (2**n - 1)] += c
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="dry run on a local fake backend")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--times", type=float, nargs="*", default=[0.5, 1.0])
    ap.add_argument("--shots", type=int, default=20000, help="shots per ancilla basis per time")
    ap.add_argument("--cal-shots", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    prob, evo, n = default_problem(), Trotter(0.25), 3

    from qiskit_ibm_runtime import SamplerV2
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService()
        backend = (svc.backend(args.backend) if args.backend
                   else svc.least_busy(operational=True, simulator=False, min_num_qubits=n + 1))
    print(f"backend: {backend.name}{' (FAKE, local noisy simulation)' if args.fake else ''}")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=args.seed)

    settings = UNIFORM.settings(n)
    pubs, keys, resources, phys_all = [], [], {}, set()
    for t in args.times:
        for a in "XY":
            tbody = pm.run(body_circuit(prob, t, a, evo))
            phys = tbody.layout.final_index_layout()[: n + 1]
            phys_all.add(tuple(phys))
            ops = tbody.count_ops()
            resources[f"t={t},{a}"] = {"two_qubit": int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))),
                                       "depth": tbody.depth(), "physical_qubits": list(map(int, phys))}
            alloc = rng.multinomial(args.shots, [p for _, p in settings])
            for (s, _), k in zip(settings, alloc):
                if k:
                    pubs.append((measured(tbody, phys, s, n), None, int(k)))
                    keys.append(("data", t, a, s))
    for k, v in resources.items():
        print(f"  {k}: {v['two_qubit']} two-qubit gates, depth {v['depth']}, qubits {v['physical_qubits']}")
    if len(phys_all) != 1:
        print(f"  final layouts differ across bodies {phys_all}; calibrating each one")
    width = backend.num_qubits
    for phys in sorted(phys_all):
        for prep in (0, 1):
            pubs.append((calibration(width, phys, n, prep), None, args.cal_shots))
            keys.append(("cal", phys, prep, None))
    total = sum(p[2] for p in pubs)
    print(f"submitting {len(pubs)} circuits, {total} shots total")

    sampler = SamplerV2(mode=backend)
    job = sampler.run(pubs)
    job_id = job.job_id()
    print(f"job id: {job_id} - waiting")
    res = job.result()

    conf = {}
    for (kind, *rest), r in zip(keys, res):
        if kind == "cal":
            phys, prep, _ = rest
            h = to_hist(r.data.c.get_counts(), n).reshape(-1)
            flips = np.zeros(n + 1)
            for v, c in enumerate(h):
                for q in range(n + 1):
                    if ((v >> q) & 1) != prep:
                        flips[q] += c / args.cal_shots
            conf.setdefault(phys, [None, None])[prep] = flips
    confusion = {phys: [np.array([[1 - f0[q], f1[q]], [f0[q], 1 - f1[q]]]) for q in range(n + 1)]
                 for phys, (f0, f1) in conf.items()}

    data = {t: TimeData(t=t, counts={}) for t in args.times}
    layout_of = {}
    for (kind, *rest), r in zip(keys, res):
        if kind != "data":
            continue
        t, a, s = rest
        data[t].counts[(a, s)] = to_hist(r.data.c.get_counts(), n)
        layout_of[t] = tuple(resources[f"t={t},{a}"]["physical_qubits"])
    raw = estimate(prob, list(data.values()), UNIFORM)
    rem = [estimate(prob, mitigate([data[t]], confusion[layout_of[t]]), UNIFORM)[0] for t in args.times]

    psi = prob.psi
    out = {"backend": backend.name, "fake": args.fake, "job_id": job_id, "resources": resources,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": []}
    print(f"\n{'t':>4s} {'quantity':10s} {'reference':>9s} {'raw':>17s} {'REM':>17s}")
    for t, r0, r1 in zip(args.times, raw, rem):
        S = np.linalg.matrix_power(trotter_step_unitary(prob.H, evo.delta), evo.steps(t))
        g = psi.conj() @ S @ psi
        refs = {"Re g": g.real, "Im g": g.imag, "<H>": prob.exact_expectation("H"),
                "<Mz>": prob.exact_expectation("Mz"),
                "Re <Mz U>": (psi.conj() @ prob.observables["Mz"].to_matrix() @ S @ psi).real}
        get = {"Re g": lambda r: (r["signal"].value.real, r["signal"].stderr.real),
               "Im g": lambda r: (r["signal"].value.imag, r["signal"].stderr.imag),
               "<H>": lambda r: (r["marg:H"].value, r["marg:H"].stderr),
               "<Mz>": lambda r: (r["marg:Mz"].value, r["marg:Mz"].stderr),
               "Re <Mz U>": lambda r: (r["joint:Mz"].value.real, r["joint:Mz"].stderr.real)}
        for q, ref in refs.items():
            (vr, er), (vm, em) = get[q](r0), get[q](r1)
            print(f"{t:4.1f} {q:10s} {ref:9.4f} {vr:9.4f}+-{er:.4f} {vm:9.4f}+-{em:.4f}")
            out["rows"].append({"t": t, "quantity": q, "reference": float(ref),
                                "raw": [float(vr), float(er)], "rem": [float(vm), float(em)]})
    name = "hardware_fake.json" if args.fake else "hardware.json"
    json.dump(out, open(OUT / name, "w"), indent=2)
    print("saved", OUT / name)


if __name__ == "__main__":
    main()
