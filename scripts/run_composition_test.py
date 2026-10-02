#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_composition_test.py -- prime families against their sieve models S7-G(z) (Table IV, Fig. 3).

For each family (primes, twin, SG, Landau) and each arm (True, S1, S7G_z for z = 1,3,5,7,
Poisson), exact ideal-chain readouts with alpha = 0.15 * 9.52 / <g>_family:
    Rg2, C_n = <R_ee^2>/(N-1), mean bond cosine, curvature energy per bead <k(1-ell)>,
    rho_id = Spearman(k, P_i), fractions of gaps divisible by 4 and by 6,
and the composition distance: KS distance between bead-stiffness distributions of True and
the arm, minus the KS distance between two realisations of the arm (KS excess).
z-scores are (True - mean_arm)/sd_arm over NREAL realisations.

Usage:  python run_composition_test.py [--N 1000000] [--nreal 40] [--Q 20]
Output: ../data/p3_composition_results.csv
"""
import os, sys, csv, time, argparse
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import core                      # noqa: E402
import sieve_models as sm        # noqa: E402

OUT = os.path.join(HERE, "..", "data", "p3_composition_results.csv")
ZS = (1, 3, 5, 7)
KEYS = ["Rg2", "Cn", "mean_cos", "curv_E", "rho_id", "frac_g4", "frac_g6"]
SEEDS = {"primes": 11, "twin": 12, "SG": 13, "Landau": 14}


def readouts(anchors, N, alpha):
    g, ia = core.gap_field(anchors, N)
    k = core.kappa_pdtp(g, ia, alpha=alpha)
    pr = core.ideal_profile(k, 1.0); ell = pr["ell"]; gaps = np.diff(anchors)
    return k, dict(Rg2=pr["Rg2"], Cn=pr["Ree2"] / (N - 1), mean_cos=float(ell[1:-1].mean()),
                   curv_E=float(np.mean(k[1:-1] * (1.0 - ell[1:-1]))), rho_id=core.spearman(k, pr["P"]),
                   frac_g4=float(np.mean(gaps % 4 == 0)), frac_g6=float(np.mean(gaps % 6 == 0)))


def ks(a, b, rng, sub=200_000):
    a = a if len(a) <= sub else rng.choice(a, sub, replace=False)
    b = b if len(b) <= sub else rng.choice(b, sub, replace=False)
    return float(stats.ks_2samp(a, b).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--nreal", type=int, default=40)
    ap.add_argument("--Q", type=int, default=20)
    ap.add_argument("--out", default=OUT)
    a_ = ap.parse_args(); N, NR, Q = a_.N, a_.nreal, a_.Q
    ip = sm.sieve(2 * N + 4)
    rows = []
    for fam in sm.FAMILIES:
        t0 = time.time(); rng = np.random.default_rng(SEEDS[fam])
        A = sm.family(fam, N, ip)
        alpha = core.ALPHA * 9.52 / np.diff(A).mean()
        edges, cnt = sm.bins_from_family(A, N, Q)
        kT, rT = readouts(A, N, alpha)
        print(f"\n[{fam}] N={N} members={len(A)} alpha={alpha:.3g} bins={len(cnt)}", flush=True)
        arms = {"S1": lambda r: sm.shuffle(A, r)}
        for z in ZS:
            c = sm.candidates(fam, N, z, ip)
            arms[f"S7G_z{z}"] = (lambda c_: (lambda r: sm.binned_draw(c_, edges, cnt, r)))(c)
        allint = np.arange(2, N)
        arms["Poisson"] = lambda r: sm.binned_draw(allint, edges, cnt, r)
        rows.append(dict(family=fam, arm="True", n_anchor=len(A), alpha=alpha, **{f"{k}_mean": rT[k] for k in KEYS},
                         **{f"{k}_sd": 0.0 for k in KEYS}, **{f"z_{k}": 0.0 for k in KEYS},
                         KS_to_true=0.0, KS_self=0.0, KS_excess=0.0))
        print(f"  {'arm':>9} " + " ".join(f"{k:>9}" for k in KEYS) + f" {'KS-excess':>10}", flush=True)
        for name, gen in arms.items():
            vals, kss, ks0, nanc, prev = {k: [] for k in KEYS}, [], [], [], None
            for _ in range(NR):
                anc = gen(rng); nanc.append(len(anc))
                kk, r = readouts(anc, N, alpha)
                for k in KEYS: vals[k].append(r[k])
                kss.append(ks(kT, kk, rng))
                if prev is not None: ks0.append(ks(prev, kk, rng))
                prev = kk
            m = {k: float(np.mean(v)) for k, v in vals.items()}
            s = {k: float(np.std(v, ddof=1)) for k, v in vals.items()}
            zz = {k: (rT[k] - m[k]) / s[k] if s[k] > 0 else float("nan") for k in KEYS}
            rows.append(dict(family=fam, arm=name, n_anchor=int(np.mean(nanc)), alpha=alpha,
                             **{f"{k}_mean": m[k] for k in KEYS}, **{f"{k}_sd": s[k] for k in KEYS},
                             **{f"z_{k}": zz[k] for k in KEYS}, KS_to_true=float(np.mean(kss)),
                             KS_self=float(np.mean(ks0)), KS_excess=float(np.mean(kss) - np.mean(ks0))))
            print(f"  {name:>9} " + " ".join(f"{zz[k]:>+9.2f}" for k in KEYS) + f" {rows[-1]['KS_excess']:>10.4f}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a_.out), exist_ok=True)
    with open(a_.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nwritten", a_.out)


if __name__ == "__main__":
    main()
