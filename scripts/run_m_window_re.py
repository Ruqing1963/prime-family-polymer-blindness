#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_m_window_re.py -- the drift-free window [1e8, 1e8 + 1e4) in the Langevin PDTP model
(Table I, Fig. 1).

Default: statistics from the released CSVs
    data/m_window_results.csv         per-trajectory rho for True, S1 (fresh sequence per
                                      trajectory), S7 (IAAFT), S7G_z5 and its run-paired shuffle
    data/m_window_random_effects.csv  K = 10 fixed S1 sequences x m = 5 trajectories
Statistics: Welch tests (naive), one-way ANOVA variance decomposition of the fixed-sequence
class, and the sequence-level statistic
    Z_seq = (rho_T - mu) / sqrt(sigma_seq^2 + s_T^2/n_T + Var mu),  Var mu = (sigma_seq^2 + sigma_th^2/m)/K.
--rerun regenerates the K x m random-effects trajectories with the Langevin engine in
code/m_window_driver.py (alpha = 0.15 ln(2e4)/ln(M + N/2), 5000 annealing steps; about 15 min
on 12 cores).  Output: data/m_window_summary.txt
"""
import os, sys, csv, math, time, zlib, argparse
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code"); DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, CODE)
MAIN = os.path.join(DATA, "m_window_results.csv"); RE = os.path.join(DATA, "m_window_random_effects.csv")
M, N, PAD, STEPS = 100_000_000, 10_000, 2_000, 5_000


def load_main():
    """rho per arm, ordered by run index (needed for the run-paired S7G / S7G_S1 contrast)."""
    d = {}
    with open(MAIN, newline="") as f:
        for r in csv.DictReader(f):
            d.setdefault(r["arm"], {})[int(r["run"])] = float(r["spearman_rho"])
    return {k: np.array([v[i] for i in sorted(v)]) for k, v in d.items()}


def load_re(K, m):
    d = {}
    with open(RE, newline="") as f:
        for r in csv.DictReader(f):
            d[(int(r["seq_id"]), int(r["run"]))] = float(r["spearman_rho"])
    return np.array([[d[(k, r)] for r in range(m)] for k in range(K) if all((k, r) in d for r in range(m))])


def random_effects(tab, T):
    K, m = tab.shape; means = tab.mean(1); mu = tab.mean()
    msb = m * np.sum((means - mu) ** 2) / (K - 1); msw = np.sum((tab - means[:, None]) ** 2) / (K * (m - 1))
    F = msb / msw; pF = 1 - stats.f.cdf(F, K - 1, K * (m - 1))
    s2 = max(0.0, (msb - msw) / m); var_mu = (s2 + msw / m) / K
    Z = (T.mean() - mu) / math.sqrt(s2 + T.var(ddof=1) / len(T) + var_mu)
    return dict(K=K, m=m, mu=mu, sigma_seq=math.sqrt(s2), sigma_th=math.sqrt(msw), F=F, p_anova=pF,
                Z_seq=Z, p_seq=2 * stats.t.sf(abs(Z), K - 1), n_above=int(np.sum(means > T.mean())), means=means)


def rerun(K, m, workers):
    import m_window_driver as drv, s7_generators as gen
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = drv.build_context(M, N, PAD); alpha = drv.ALPHA * drv.ALPHA_REF_LNX / math.log(M + N / 2)
    jobs = [(ctx, k, r, alpha) for k in range(K) for r in range(m)]
    with ProcessPoolExecutor(max_workers=workers) as ex, open(RE, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "seq_id", "run", "seed_seq", "seed_traj", "spearman_rho", "dR", "n_anchor_inside", "mean_gap", "seconds"])
        for fu in as_completed([ex.submit(_re_job, j) for j in jobs]):
            w.writerow(fu.result()); f.flush()


def _re_job(job):
    import m_window_driver as drv, s7_generators as gen
    ctx, k, r, alpha = job; t0 = time.time()
    s_seq = zlib.crc32(f"Mwin|S1fix|seq{k}".encode()) & 0xFFFFFFFF
    s_traj = zlib.crc32(f"Mwin|S1fix|seq{k}|run{r}".encode()) & 0xFFFFFFFF
    inner = gen.generate_s1(ctx["inner"], np.random.default_rng([s_seq, 1]))
    anchors = np.concatenate(([ctx["left"]], inner, [ctx["right"]]))
    g, ia = drv.gap_field(anchors - ctx["M"], ctx["N"]); kb = drv.stiffness_from_gap(g, ia, alpha)
    rng = np.random.default_rng([s_traj, 2])
    x = drv.run_langevin(drv.init_chain(ctx["N"], rng), kb, STEPS, rng)
    rho, dR = drv.observables(x, kb)
    return ["S1fix", k, r, s_seq, s_traj, f"{rho:.6f}", f"{dR:.6f}", len(inner), float(np.diff(inner).mean()), time.time() - t0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--rerun", action="store_true"); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.rerun: rerun(a.K, a.m, a.workers)
    arms = load_main(); T, S1 = arms["True"], arms["S1"]
    L = ["Drift-free window [1e8, 1e8+1e4), Langevin PDTP, alpha = 0.0806, 5000 steps", ""]
    L.append(f"  {'arm':>11} {'n':>3} {'rho':>18} {'Welch p vs True':>16}")
    for arm in ("True", "S1", "S7", "S7G_z5", "S7G_S1_z5"):
        if arm in arms:
            v = arms[arm]; p = stats.ttest_ind(v, T, equal_var=False).pvalue if arm != "True" else float("nan")
            L.append(f"  {arm:>11} {len(v):>3} {v.mean():+.4f} +- {v.std(ddof=1)/math.sqrt(len(v)):.4f} {p:>16.3g}")
    if "S7G_z5" in arms and "S7G_S1_z5" in arms:
        d = arms["S7G_z5"] - arms["S7G_S1_z5"]
        L.append(f"  paired ordering effect inside S7-G(z=5): {d.mean():+.4f} (p = {stats.ttest_1samp(d, 0).pvalue:.2f})")
    re = random_effects(load_re(a.K, a.m), T)
    L += ["", f"  fixed-S1 class: K = {re['K']} x m = {re['m']}",
          f"    mu = {re['mu']:+.4f}  sigma_seq = {re['sigma_seq']:.4f}  sigma_th = {re['sigma_th']:.4f}  ANOVA F = {re['F']:.2f} (p = {re['p_anova']:.3f})",
          "    sequence means: " + " ".join(f"{x:+.3f}" for x in re["means"]),
          f"    True - mu = {T.mean()-re['mu']:+.4f}   Z_seq = {re['Z_seq']:+.2f}  (t, {re['K']-1} dof, p = {re['p_seq']:.3f})"
          f"   sequence means above True: {re['n_above']}/{re['K']}",
          f"    naive Welch True vs fresh S1: p = {stats.ttest_ind(T, S1, equal_var=False).pvalue:.3g}"]
    txt = "\n".join(L); print(txt)
    with open(os.path.join(DATA, "m_window_summary.txt"), "w") as f: f.write(txt + "\n")


if __name__ == "__main__":
    main()
