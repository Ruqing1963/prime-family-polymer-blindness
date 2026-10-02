#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sq_channel_test.py -- do exact ideal-chain readouts see the mod-q congruence channel?
(Tables II and III, Fig. 2.)

Part 1  positional readout rho_id (N = 4e4): for each family, True vs the S1 ensemble, and the
        kill / keep jitter contrasts at q = 3, 4, 6.                 -> rho_channel_results.csv
Part 2  jitter audit (primes, N = 4e4): gap-multiset KS distance and gap-variance change of
        the kill, keep (v1) and variance-matched keep arms at q = 6, 7, with the Bragg power
        at a/q certifying what each arm did to the channel.          -> jitter_audit.csv
Part 3  scattering function S(q) in the Debye-Gauss closure of the exact ideal chain: primes
        N = 6000 and twin primes N = 12000; arms True, S1 ensemble, kill / keep / keep_vm at
        the heavy modulus 6 and the light modulus 7.                 -> sq_channel_results.csv
Part 3 is O(N^2) and takes ~30 min; --quick shrinks it to a few minutes.

Usage:  python run_sq_channel_test.py [--parts 123] [--quick]
"""
import os, sys, csv, time, argparse
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import core                      # noqa: E402
import sieve_models as sm        # noqa: E402
import debye_gauss as dg         # noqa: E402

DATA = os.path.join(HERE, "..", "data")


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", path, flush=True)


def part1(nreal):
    N = 40_000; ip = sm.sieve(2 * N + 4); rows = []
    gbar_ref = np.diff(sm.family("primes", N, ip)).mean()
    print(f"\nPart 1: rho_id, N={N}, {nreal} realisations per arm")
    for fam in sm.FAMILIES:
        rng = np.random.default_rng(100 + sm.FAMILIES.index(fam))
        a = sm.family(fam, N, ip); alpha = core.ALPHA * gbar_ref / np.diff(a).mean()
        rho = lambda anc: dg.rho_ideal(dg.kappa_from_anchors(anc, N, alpha))
        rT = rho(a); rS1 = np.array([rho(sm.shuffle(a, rng)) for _ in range(nreal)])
        row = dict(family=fam, n_anchor=len(a), alpha=alpha, rho_true=rT, rho_S1_mean=rS1.mean(),
                   rho_S1_sd=rS1.std(ddof=1), z_true=(rT - rS1.mean()) / rS1.std(ddof=1))
        for q in (3, 4, 6):
            for arm in ("kill", "keep"):
                v = np.mean([rho(sm.jitter(a, q, rng, arm)) for _ in range(nreal)])
                row[f"d_{arm}{q}"] = v - rT
        rows.append(row)
        print(f"  {fam:>7} n={len(a):5d} z(True vs S1)={row['z_true']:+.1f}  " +
              "  ".join(f"q={q}: kill {row[f'd_kill{q}']:+.4f} keep {row[f'd_keep{q}']:+.4f}" for q in (3, 4, 6)), flush=True)
    write(os.path.join(DATA, "rho_channel_results.csv"), rows)


def part2(reps=20):
    N = 40_000; P = np.where(sm.sieve(N))[0]; g = np.diff(P); rows = []
    rng = np.random.default_rng(200)
    print(f"\nPart 2: jitter audit, primes N={N}, {reps} realisations")
    print(f"  true Bragg power: a/6 {sm.bragg_power(P, 6):.2e}, a/7 {sm.bragg_power(P, 7):.2e}")
    for q in (6, 7):
        for arm in ("kill", "keep", "keep_vm"):
            ks, dv, bq = [], [], []
            for _ in range(reps):
                x = sm.jitter(P, q, rng, arm); gg = np.diff(x)
                ks.append(stats.ks_2samp(g, gg).statistic); dv.append(abs(gg.var() - g.var()) / g.var())
                bq.append(sm.bragg_power(x, q))
            rows.append(dict(q=q, arm=arm, KS_gaps=np.mean(ks), dVar_rel=np.mean(dv), bragg_aq=np.mean(bq),
                             bragg_true=sm.bragg_power(P, q)))
            print(f"  q={q} {arm:>8}: KS(gaps)={np.mean(ks):.3f}  |dVar|/Var={np.mean(dv):.3f}  Bragg a/q={np.mean(bq):.1e}", flush=True)
    write(os.path.join(DATA, "jitter_audit.csv"), rows)


def part3(quick):
    rows = []
    plan = (("primes", 3000, 4), ("twin", 6000, 4)) if quick else (("primes", 6000, 12), ("twin", 12000, 8))
    print(f"\nPart 3: S(q), {'quick' if quick else 'full'} mode")
    for fam, N, nreal in plan:
        ip = sm.sieve(2 * N + 4); a = sm.family(fam, N, ip); alpha = core.ALPHA * 9.52 / np.diff(a).mean()
        rng = np.random.default_rng(300 + sm.FAMILIES.index(fam)); t0 = time.time()
        S = lambda anc: dg.S_of_q(dg.kappa_from_anchors(anc, N, alpha))
        S_true = S(a)
        S1 = np.array([S(sm.shuffle(a, rng)) for _ in range(nreal)]); m, s = S1.mean(0), S1.std(0, ddof=1)
        arms = {"True": np.array([S_true])}
        for q in (6, 7):
            for arm in ("kill", "keep", "keep_vm"):
                arms[f"{arm}{q}"] = np.array([S(sm.jitter(a, q, rng, arm)) for _ in range(nreal)])
        print(f"  [{fam}] N={N} anchors={len(a)} ({time.time()-t0:.0f}s)")
        print(f"     {'arm':>9} {'max|dS/S|':>10} {'max|dS|/sd':>11} {'mean dS/S':>10}")
        for lab, arr in arms.items():
            rel, zq = dg.relative_deviation(arr.mean(0), m, s)
            print(f"     {lab:>9} {np.abs(rel).max():>10.4f} {np.abs(zq).max():>11.1f} {rel.mean():>+10.4f}", flush=True)
            for i, qq in enumerate(dg.Q_DEFAULT):
                rows.append(dict(family=fam, N=N, arm=lab, q=qq, S=arr.mean(0)[i], S1_mean=m[i], S1_sd=s[i], rel=rel[i], z=zq[i]))
        for arm in ("kill", "keep", "keep_vm"):
            dd = (arms[f"{arm}6"].mean(0) - arms[f"{arm}7"].mean(0)) / m
            print(f"     double difference {arm}: [Delta(6) - Delta(7)] max |.| = {np.abs(dd).max():.4f}", flush=True)
    write(os.path.join(DATA, "sq_channel_results.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="123"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nreal", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if "1" in a.parts: part1(4 if a.quick else a.nreal)
    if "2" in a.parts: part2(4 if a.quick else 20)
    if "3" in a.parts: part3(a.quick)


if __name__ == "__main__":
    main()
