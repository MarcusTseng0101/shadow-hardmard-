"""Shadow-enhanced Hadamard test: circuits, Aer sampling and estimators.

Circuit (system = qubits 0..n-1, ancilla = qubit n):

    ancilla : |0> -H-- * --[S^dag if Y]-H- measure  -> s = +/-1
    system  : |psi> --[U]--[random Pauli basis]- measure -> classical shadow

After the controlled evolution the joint state is (|0>|psi> + |1>U|psi>)/sqrt2.  Measuring the
ancilla in the X (Y) basis and the system with a randomized Pauli measurement gives, for any
system observable O (see README for the derivation)

    E[s]              = Re (X) / Im (Y) of  g(t)           = <psi|U|psi>
    E[s * o_hat]      = Re (X) / Im (Y) of  <psi| O U |psi>
    E[o_hat]          = ( <O>_psi + <O>_{U psi} ) / 2      (= <O>_psi if [O, H] = 0)

where o_hat is the (biased) classical-shadow estimate of O from one system snapshot.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from .model import Problem


# --------------------------------------------------------------------------- scheme
@dataclass(frozen=True)
class PauliShadowScheme:
    """Locally randomized Pauli measurements with basis probabilities (pX, pY, pZ).

    (1/3, 1/3, 1/3) is the standard classical-shadow protocol of Huang, Kueng, Preskill.
    Biasing towards Z lowers the variance of diagonal observables (symmetry projectors, ZZ terms)."""
    pX: float = 1 / 3
    pY: float = 1 / 3
    pZ: float = 1 / 3

    @property
    def probs(self) -> dict[str, float]:
        tot = self.pX + self.pY + self.pZ
        return {"X": self.pX / tot, "Y": self.pY / tot, "Z": self.pZ / tot}

    def settings(self, n: int):
        """All n-qubit basis settings with non-zero probability; setting[i] is the basis of qubit i."""
        p = self.probs
        out = []
        for s in itertools.product("XYZ", repeat=n):
            prob = float(np.prod([p[b] for b in s]))
            if prob > 0:
                out.append((s, prob))
        return out

    def supports(self, obs: SparsePauliOp) -> bool:
        """An observable is estimable iff every Pauli it contains can actually be measured."""
        p = self.probs
        return all(p[c] > 0 for label in obs.paulis.to_labels() for c in label if c != "I")

    def __str__(self):
        p = self.probs
        return f"Pauli shadows (pX={p['X']:.2f}, pY={p['Y']:.2f}, pZ={p['Z']:.2f})"


UNIFORM = PauliShadowScheme()


def _signs(n: int) -> np.ndarray:
    """signs[i, k] = +1/-1 eigenvalue of qubit i in outcome index k (bit 0 -> +1)."""
    k = np.arange(2**n)
    return np.array([1 - 2 * ((k >> i) & 1) for i in range(n)], dtype=float)


def shadow_table(obs: SparsePauliOp, setting, scheme: PauliShadowScheme) -> np.ndarray:
    """Single-snapshot shadow estimate of `obs` for every outcome k of basis `setting`.

    For a Pauli string P the biased-shadow estimator is prod_{i in supp P} [b_i == P_i] s_i / p_{P_i};
    it is unbiased because each basis is picked with probability p_b."""
    n = obs.num_qubits
    signs = _signs(n)
    p = scheme.probs
    table = np.zeros(2**n)
    for label, coeff in zip(obs.paulis.to_labels(), obs.coeffs):
        term = np.full(2**n, float(np.real(coeff)))
        for i in range(n):
            P = label[n - 1 - i]  # qiskit labels are little-endian
            if P == "I":
                continue
            if setting[i] != P:
                term = None
                break
            term = term * signs[i] / p[P]
        if term is not None:
            table += term
    return table


# --------------------------------------------------------------------------- circuits
def controlled_evolution(problem: Problem, t: float) -> UnitaryGate:
    """|0><0| (x) 1 + |1><1| (x) e^{-iHt}, ancilla as the most significant qubit (exact unitary)."""
    D = 2**problem.n
    CU = np.zeros((2 * D, 2 * D), dtype=complex)
    CU[:D, :D] = np.eye(D)
    CU[D:, D:] = problem.U(t)
    return UnitaryGate(CU, label=f"c-U({t:.2f})")


def _rotate_to(qc: QuantumCircuit, basis: str, q):
    if basis == "X":
        qc.h(q)
    elif basis == "Y":
        qc.sdg(q)
        qc.h(q)


def hadamard_circuit(problem: Problem, t: float, anc_basis: str, setting, measure=True):
    n = problem.n
    sysq, anc = QuantumRegister(n, "sys"), QuantumRegister(1, "anc")
    qc = QuantumCircuit(sysq, anc)
    qc.compose(problem.prep, qubits=list(sysq), inplace=True)
    qc.h(anc[0])
    qc.append(controlled_evolution(problem, t), list(sysq) + [anc[0]])
    _rotate_to(qc, anc_basis, anc[0])
    for i, b in enumerate(setting):
        _rotate_to(qc, b, sysq[i])
    if measure:
        c = ClassicalRegister(n + 1, "c")
        qc.add_register(c)
        qc.measure(list(sysq) + [anc[0]], c)
    return qc


# --------------------------------------------------------------------------- data taking
@dataclass
class TimeData:
    """Outcome weights for one evolution time: counts[(anc_basis, setting)] has shape (2, 2**n),
    axis 0 = ancilla bit (0 -> s=+1), axis 1 = system outcome index."""
    t: float
    counts: dict


def run_experiment(problem: Problem, times, shots_per_basis: int, scheme: PauliShadowScheme = UNIFORM,
                   seed: int | None = None, mode: str = "aer") -> list[TimeData]:
    """Take shadow-Hadamard data at each time. `shots_per_basis` shots go to each ancilla basis (X, Y).

    mode="aer"   : Qiskit Aer sampling, basis settings drawn at random for every shot.
    mode="exact" : exact outcome probabilities (infinite shots) - used to verify the estimators."""
    rng = np.random.default_rng(seed)
    n = problem.n
    settings = scheme.settings(n)
    data = [TimeData(t=float(t), counts={}) for t in times]
    pubs, keys = [], []
    for td in data:
        for a in "XY":
            if mode == "exact":
                for s, prob in settings:
                    qc = hadamard_circuit(problem, td.t, a, s, measure=False)
                    pr = Statevector.from_instruction(qc).probabilities()  # index = anc*2^n + sys
                    td.counts[(a, s)] = prob * pr.reshape(2, 2**n)
                continue
            alloc = rng.multinomial(shots_per_basis, [p for _, p in settings])
            for (s, _), k in zip(settings, alloc):
                if k:
                    pubs.append((hadamard_circuit(problem, td.t, a, s), None, int(k)))
                    keys.append((td, a, s))
    if pubs:
        # One Aer run per circuit, each with its own seed. Batching the pubs of one Aer SamplerV2
        # call with a fixed seed reuses the same random stream for every circuit, which correlates
        # the ancilla outcomes across basis settings and inflates the variance ~sqrt(#settings)-fold.
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        for (qc, _, shots), (td, a, s) in zip(pubs, keys):
            counts = sim.run(qc, shots=shots, seed_simulator=int(rng.integers(2**31))).result().get_counts()
            arr = np.zeros((2, 2**n))
            for key, c in counts.items():
                v = int(key, 2)
                arr[(v >> n) & 1, v & (2**n - 1)] += c
            td.counts[(a, s)] = arr
    return data


# --------------------------------------------------------------------------- estimators
@dataclass
class Estimate:
    value: complex | float
    stderr: complex | float


def _mean_err(sum_x, sum_x2, w):
    mean = sum_x / w
    var = max(sum_x2 / w - mean**2, 0.0)
    # w is a shot count for sampled data, a probability (=1) for exact data -> no error bar
    return mean, (np.sqrt(var / w) if w > 1.5 else 0.0)


def estimate(problem: Problem, data: list[TimeData], scheme: PauliShadowScheme = UNIFORM,
             names=None) -> list[dict]:
    """Per time: signal g(t), joint <psi|O U|psi> and marginal (<O>+<O>_U)/2 for each observable.

    Observables that the scheme cannot measure (e.g. XX terms with pX = 0) are reported as NaN."""
    n = problem.n
    names = list(problem.observables) if names is None else names
    ok = {nm: scheme.supports(problem.observables[nm]) for nm in names}
    tables = {}
    s_anc = np.array([1.0, -1.0])[:, None]
    out = []
    for td in data:
        acc = {a: {"w": 0.0, "s": 0.0, "s2": 0.0} for a in "XY"}
        J = {a: {nm: [0.0, 0.0] for nm in names} for a in "XY"}  # sum(s v), sum(v^2)
        M = {nm: [0.0, 0.0] for nm in names}                      # sum(v), sum(v^2) over both bases
        for (a, s), C in td.counts.items():
            acc[a]["w"] += C.sum()
            acc[a]["s"] += (C * s_anc).sum()
            acc[a]["s2"] += C.sum()
            if s not in tables:
                tables[s] = {nm: shadow_table(problem.observables[nm], s, scheme) for nm in names}
            for nm in names:
                v = tables[s][nm][None, :]
                J[a][nm][0] += (C * s_anc * v).sum()
                J[a][nm][1] += (C * v**2).sum()
                M[nm][0] += (C * v).sum()
                M[nm][1] += (C * v**2).sum()
        row = {"t": td.t}
        re, re_e = _mean_err(acc["X"]["s"], acc["X"]["s2"], acc["X"]["w"])
        im, im_e = _mean_err(acc["Y"]["s"], acc["Y"]["s2"], acc["Y"]["w"])
        row["signal"] = Estimate(re + 1j * im, re_e + 1j * im_e)
        w_all = acc["X"]["w"] + acc["Y"]["w"]
        for nm in names:
            if not ok[nm]:
                row[f"joint:{nm}"] = Estimate(np.nan + 1j * np.nan, np.nan + 1j * np.nan)
                row[f"marg:{nm}"] = Estimate(np.nan, np.nan)
                continue
            re, re_e = _mean_err(*J["X"][nm], acc["X"]["w"])
            im, im_e = _mean_err(*J["Y"][nm], acc["Y"]["w"])
            row[f"joint:{nm}"] = Estimate(re + 1j * im, re_e + 1j * im_e)
            row[f"marg:{nm}"] = Estimate(*_mean_err(*M[nm], w_all))
        out.append(row)
    return out


def pooled_marginal(rows: list[dict], name: str) -> Estimate:
    """Inverse-variance average of the marginal over all times - valid for conserved O, where every
    time slice estimates the same number <O>_psi. This is the 'free' energy / symmetry readout."""
    vals = np.array([r[f"marg:{name}"].value for r in rows], dtype=float)
    errs = np.array([r[f"marg:{name}"].stderr for r in rows], dtype=float)
    if np.all(errs == 0):
        return Estimate(float(np.mean(vals)), 0.0)
    w = 1 / errs**2
    return Estimate(float(np.sum(w * vals) / np.sum(w)), float(1 / np.sqrt(np.sum(w))))
