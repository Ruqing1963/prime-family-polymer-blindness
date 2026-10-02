#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper3_m_window_driver.py -- Paper 3: PDTP folding on the prime window [M, M+N).

Design (approved blueprint)
---------------------------
  window   : beads i = 0..N-1  <->  integers M+i, default M = 1e8, N = 1e4
             (the 1/ln i gradient across the window is ~5e-6: S6' and the
             index-partial-correlation control are not needed here)
  arms     : True      the primes of the window
             S1        Paper-1 shuffle of the interior gaps (exact multiset)
             S7        IAAFT surrogate: exact multiset + spectrum of the detrended gaps
             S7G_z*    Granville-Cramer model at sieve depth z (default z = 5, 11, 31, 100)
             S7G_z*_S1 (optional, --s7g-shuffle-control) the shuffle of each S7-G
                       realisation, run-paired, so that ordering and composition can be
                       separated inside the S7-G arm exactly as True is separated from S1
  protocol : the PDTP force field, overdamped Langevin integrator, displacement cap and
             geometric annealing of Papers 1-2, copied verbatim (Chen 2026, Zenodo
             10.5281/zenodo.22020688); 5000 steps, n = 40 per arm
  readout  : Spearman rho(stiffness, radial distance from the centre of mass) and the
             tail-weighted core-shell separation Delta R (p80 - p20), as in Papers 1-2
  boundary : the gap field is painted from the anchors that bracket the window (the true
             primes just outside [M, M+N) for True/S1/S7; the model's own for S7-G), so the
             chain ends carry real gaps instead of the [0, N) end conventions of Paper 1

Usage
-----
  python3 paper3_m_window_driver.py                       # full experiment, resumes from CSV
  python3 paper3_m_window_driver.py --workers 8           # parallel trajectories
  python3 paper3_m_window_driver.py --maxrun 20           # do <= 20 new trajectories, then stop
  python3 paper3_m_window_driver.py --smoke               # tiny end-to-end check (~1 min)
  python3 paper3_m_window_driver.py --certificate-only    # two-point certificate table only
  python3 paper3_m_window_driver.py --summarize-only      # statistics from the existing CSV
  python3 paper3_m_window_driver.py --alpha-rescale       # alpha * ln(2e4)/ln(M): keep the
                                                          # stiffness contrast of the Letter
Outputs (in --outdir, default ./paper3_out)
  results.csv       one row per trajectory (resumable)
  certificate.csv   two-point diagnostics of the sequences actually folded, vs Hardy-Littlewood
  summary.txt       per-arm statistics, Welch tests, Cohen d, Delta rho(z)

Dependencies: numpy, scipy, and s7_generators.py in the same directory.
"""
from __future__ import annotations
import os
# Pin BLAS/OpenMP to one thread per process *before* numpy is imported.  Each trajectory
# already occupies one worker process; without this, MKL/OpenBLAS oversubscribe the CPU
# and inflate per-worker memory -- the failure mode seen with --workers 8 on a laptop
# (numpy MemoryError for a 234 KiB array = the machine ran out of RAM, not the algorithm).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse, math, json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s7_generators as gen

# =============================================================================
# PDTP force field and protocol constants -- verbatim from Papers 1-2
# =============================================================================
B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003
BASE_SEED = 20260903

ALPHA_REF_LNX = math.log(2.0e4)   # mid-chain of the Letter's decisive N = 4e4 chain (for --alpha-rescale)


def _scatter_add(F, idx, vals):
    N = F.shape[0]
    for c in range(3):
        F[:, c] += np.bincount(idx, weights=vals[:, c], minlength=N)


def compute_forces(r, kbend):
    """Bond + gap-driven bending + WCA (kd-tree, force cap) + weak confinement.  Verbatim."""
    N = r.shape[0]; F = np.zeros_like(r)
    d = r[1:] - r[:-1]; L = np.linalg.norm(d, axis=1) + 1e-12
    fb = (K_BOND * (L - B0) / L)[:, None] * d; F[0:N - 1] += fb; F[1:N] -= fb
    a = r[1:-1] - r[:-2]; b = r[2:] - r[1:-1]
    na = np.linalg.norm(a, axis=1) + 1e-12; nb = np.linalg.norm(b, axis=1) + 1e-12
    ah, bh = a / na[:, None], b / nb[:, None]
    cos = np.clip(np.sum(ah * bh, axis=1), -1.0, 1.0); kk = kbend[1:-1][:, None]
    dda = (bh - cos[:, None] * ah) / na[:, None]; ddb = (ah - cos[:, None] * bh) / nb[:, None]
    F[0:N - 2] += -kk * dda; F[2:N] += kk * ddb; F[1:N - 1] += kk * (dda - ddb)
    tree = cKDTree(r); pairs = tree.query_pairs(RC_LJ * SIGMA, output_type="ndarray")
    if len(pairs) > 0:
        i, j = pairs[:, 0], pairs[:, 1]; keep = np.abs(i - j) > 1; i, j = i[keep], j[keep]
        for s in range(0, len(i), 4_000_000):
            ii, jj = i[s:s + 4_000_000], j[s:s + 4_000_000]
            rij = r[ii] - r[jj]; dist = np.linalg.norm(rij, axis=1) + 1e-12
            sr6 = (SIGMA / dist) ** 6
            fmag = np.clip(24 * EPS_LJ * (2 * sr6 ** 2 - sr6) / dist, 0.0, FCAP_LJ)
            fvec = (fmag / dist)[:, None] * rij
            _scatter_add(F, ii, fvec); _scatter_add(F, jj, -fvec)
    if K_CONF > 0.0:
        F += -K_CONF * (r - r.mean(axis=0))
    return F


def init_chain(N, rng, persistence=0.55):
    dirs = np.zeros((N, 3)); dv = rng.standard_normal(3); dv /= np.linalg.norm(dv); dirs[0] = dv
    for i in range(1, N):
        dv = persistence * dv + (1 - persistence) * rng.standard_normal(3)
        dv /= np.linalg.norm(dv) + 1e-12; dirs[i] = dv
    r = np.cumsum(dirs, axis=0) * B0; return r - r.mean(axis=0)


def run_langevin(r, kbend, n_steps, rng):
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1); kT = KT_HI * (KT_LO / KT_HI) ** frac
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * DT + np.sqrt(2.0 * MOBILITY * kT * DT) * rng.standard_normal(r.shape)
        dmag = np.linalg.norm(disp, axis=1); big = dmag > MAX_DISP
        if np.any(big):
            disp[big] *= (MAX_DISP / dmag[big])[:, None]
        r += disp
    return r


def observables(r, kbend):
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1)
    rho, _ = stats.spearmanr(kbend, rad)
    hi = kbend > np.percentile(kbend, 80); lo = kbend < np.percentile(kbend, 20)
    if not hi.any() or not lo.any():
        # Ties at the cap (alpha=0.15 at M~1e8 puts >20% of beads exactly at k_max) empty
        # the strict percentile sets of Paper 1 and made dR = nan.  Fall back to rank-based
        # top/bottom 20% (identical to the percentile definition when there are no ties).
        order = np.argsort(kbend, kind="stable"); k = max(1, len(kbend) // 5)
        lo = np.zeros(len(kbend), bool); hi = np.zeros(len(kbend), bool)
        lo[order[:k]] = True; hi[order[-k:]] = True
    return float(rho), float(rad[hi].mean() - rad[lo].mean())


# =============================================================================
# Window, arms, gap field
# =============================================================================
def gap_field(anchors_rel: np.ndarray, N: int):
    """Paint the gap field on beads 0..N-1 from anchors given relative to the window origin.
    The anchor array must bracket the window (first < 0, last >= N)."""
    a = np.sort(np.asarray(anchors_rel, dtype=np.int64))
    if a[0] >= 0 or a[-1] < N:
        raise ValueError("anchors must bracket the window")
    g = np.zeros(N); is_anchor = np.zeros(N, dtype=bool)
    inside = a[(a >= 0) & (a < N)]; is_anchor[inside] = True
    for x, y in zip(a[:-1], a[1:]):
        lo, hi = max(int(x) + 1, 0), min(int(y), N)
        if hi > lo:
            g[lo:hi] = y - x
    return g, is_anchor


def stiffness_from_gap(g, is_anchor, alpha):
    k = np.full(len(g), K_HINGE); comp = ~is_anchor
    k[comp] = np.minimum(K_HINGE * np.exp(alpha * g[comp]), K_MAX)
    return k


def build_context(M: int, N: int, pad: int):
    """True primes on the extended range [M-pad, M+N+pad) and the window bracket."""
    lo_ext, hi_ext = M - pad, M + N + pad
    ext = gen.segmented_primes(lo_ext, hi_ext)
    left, inner, right = gen.split_window(ext, M, M + N)
    return {"M": int(M), "N": int(N), "pad": int(pad), "lo_ext": int(lo_ext), "hi_ext": int(hi_ext),
            "ext_primes": ext, "left": int(left), "inner": inner, "right": int(right),
            "n_ext": int(len(ext))}


ARM_INDEX = {"True": 0, "S1": 1, "S7": 2}          # S7G_z -> 3+k, S7G_z_S1 -> 20+k (k = index in z list)


def arm_name(kind: str, z: int | None) -> str:
    return kind if z is None else f"{kind}_z{z}"


def arm_seed(kind: str, z: int | None, run: int, z_list) -> int:
    if kind in ARM_INDEX:
        idx = ARM_INDEX[kind]
    elif kind == "S7G":
        idx = 3 + list(z_list).index(z)
    elif kind == "S7G_S1":
        idx = 20 + list(z_list).index(z)
    else:
        raise ValueError(kind)
    return BASE_SEED + idx * 1_000_000 + int(run)


def build_anchors(kind: str, z, run: int, ctx: dict, z_list, detrend="pnt"):
    """Absolute anchor positions bracketing the window, plus sequence metadata."""
    seed = arm_seed(kind, z, run, z_list)
    rng = np.random.default_rng([seed, 1])
    meta = {"warning": ""}
    if kind == "True":
        anchors = ctx["ext_primes"]
    elif kind == "S1":
        inner = gen.generate_s1(ctx["inner"], rng)
        anchors = np.concatenate(([ctx["left"]], inner, [ctx["right"]]))
    elif kind == "S7":
        inner, info = gen.generate_s7_iaaft(ctx["inner"], rng, detrend=detrend)
        anchors = np.concatenate(([ctx["left"]], inner, [ctx["right"]]))
        meta["iaaft_iterations"] = info["iterations"]; meta["iaaft_spectral_error"] = info["spectral_error"]
    elif kind in ("S7G", "S7G_S1"):
        # the S7-G realisation is always drawn with the S7G arm's seed, so S7G_S1 is run-paired
        rng_g = np.random.default_rng([arm_seed("S7G", z, run, z_list), 1])
        anchors, info = gen.generate_s7_g(ctx["lo_ext"], ctx["hi_ext"], z, rng_g,
                                          count_mode="fixed", target_count=ctx["n_ext"])
        M, N = ctx["M"], ctx["N"]
        if not (anchors[0] < M and anchors[-1] >= M + N):        # vanishingly rare with pad ~ 100 gaps
            anchors = np.unique(np.concatenate(([ctx["left"]], anchors, [ctx["right"]])))
            meta["warning"] = "bracket completed with true bounding primes"
        meta["theta_z"] = info["theta_z"]; meta["survival_prob"] = info["survival_prob"]
        if kind == "S7G_S1":
            left, inner, right = gen.split_window(anchors, M, M + N)
            anchors = np.concatenate(([left], gen.generate_s1(inner, rng), [right]))
    else:
        raise ValueError(kind)
    M, N = ctx["M"], ctx["N"]
    inner = anchors[(anchors >= M) & (anchors < M + N)]
    gaps = np.diff(inner).astype(float)
    meta.update({"n_anchor_inside": int(len(inner)), "mean_gap": float(gaps.mean()),
                 "gap_cv2": gen.gap_cv2(gaps)})
    return anchors, meta


# =============================================================================
# One trajectory
# =============================================================================
def run_one_job(job: dict) -> dict:
    ctx = job["ctx"]; kind, z, run = job["kind"], job["z"], job["run"]
    t0 = time.time()
    anchors, meta = build_anchors(kind, z, run, ctx, job["z_list"], job["detrend"])
    g, is_anchor = gap_field(anchors - ctx["M"], ctx["N"])
    kbend = stiffness_from_gap(g, is_anchor, job["alpha"])
    frac_sat = float(np.mean(kbend >= K_MAX * (1 - 1e-12)))
    rng_dyn = np.random.default_rng([arm_seed(kind, z, run, job["z_list"]), 2])
    r = init_chain(ctx["N"], rng_dyn)
    r = run_langevin(r, kbend, job["steps"], rng_dyn)
    rho, dR = observables(r, kbend)
    return {"arm": arm_name(kind, z), "kind": kind, "z": "" if z is None else int(z), "run": int(run),
            "seed": arm_seed(kind, z, run, job["z_list"]), "spearman_rho": rho, "dR": dR,
            "n_anchor_inside": meta["n_anchor_inside"], "mean_gap": meta["mean_gap"],
            "gap_cv2": meta["gap_cv2"], "frac_beads_at_kmax": frac_sat,
            "iaaft_spectral_error": meta.get("iaaft_spectral_error", ""),
            "seconds": time.time() - t0, "warning": meta.get("warning", "")}


CSV_FIELDS = ["arm", "kind", "z", "run", "seed", "spearman_rho", "dR", "n_anchor_inside", "mean_gap",
              "gap_cv2", "frac_beads_at_kmax", "iaaft_spectral_error", "seconds", "warning"]


def load_done(path):
    done = {}
    if os.path.exists(path):
        import csv
        with open(path) as f:
            for row in csv.DictReader(f):
                done[(row["arm"], int(row["run"]))] = row
    return done


def append_row(path, row):
    import csv
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# =============================================================================
# Certificate: two-point content of the sequences actually folded, vs Hardy-Littlewood
# =============================================================================
def certificate(ctx, arms, nens, z_list, detrend, H_list=(100, 1000), cert_range=0, out_path=None):
    M, N = ctx["M"], ctx["N"]; X = M + N / 2
    rows = []; lines = []
    prime_table = gen.small_primes(max(H_list) + 10)
    hl_full = {H: gen.hl_prime_count_fano(H, X, prime_table=prime_table) for H in H_list}
    lines.append("Two-point certificate: count Fano factor Var/mean of anchors in windows of H integers")
    lines.append("(window sequences use sliding windows, stride H/4; ensemble mean +- s.e.m. over the")
    lines.append(" nens realisations that are actually folded; True over the whole window)")
    mean_gap_true = float(np.diff(ctx["inner"]).mean())
    L_list = [max(2, int(round(H / mean_gap_true))) for H in H_list]
    lines.append("Gap-coordinate companion: span-variance ratio Var(sum of L gaps)/(L Var g) at L = H/<g>"
                 " (1 = i.i.d. gaps, <1 = rigid); this is the statistic IAAFT matches directly.")
    hdr = "  %-14s" % "arm" + "".join("  H=%-5d          " % H for H in H_list) + "".join("  L=%-4d        " % L for L in L_list) + "  <g>    CV^2   note"
    lines.append(hdr)
    lines.append("  %-14s" % "HL (full)" + "".join("  %.3f            " % hl_full[H] for H in H_list) + "".join("  %-14s" % "-" for L in L_list) + "  -      -      Part-29 identity, Part-14 pair series")
    for kind, z in arms:
        name = arm_name(kind, z); fan = {H: [] for H in H_list}; svr = {L: [] for L in L_list}; cv2 = []; mg = []; sp_err = []
        reps = 1 if kind == "True" else nens
        for run in range(reps):
            anchors, meta = build_anchors(kind, z, run, ctx, z_list, detrend)
            gaps_in = np.diff(anchors[(anchors >= M) & (anchors < M + N)]).astype(float)
            for H in H_list:
                fan[H].append(gen.count_fano(anchors, M, M + N, H, stride=max(1, H // 4)))
            for L in L_list:
                svr[L].append(gen.span_variance_ratio(gaps_in, L))
            cv2.append(meta["gap_cv2"]); mg.append(meta["mean_gap"])
            if "iaaft_spectral_error" in meta:
                sp_err.append(meta["iaaft_spectral_error"])
        note = ""
        if kind == "S7":
            note = "IAAFT spectral error %.3f" % float(np.mean(sp_err))
        if kind in ("S7G", "S7G_S1"):
            note = "HL truncated at z: " + ", ".join("H=%d %.3f" % (H, gen.hl_prime_count_fano(H, X, K=z, prime_table=prime_table)) for H in H_list)
        cells = ""
        for H in H_list:
            v = np.array(fan[H], float); m = float(np.nanmean(v)); s = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            cells += "  %.3f +- %.3f   " % (m, s) if len(v) > 1 else "  %.3f (1 seq)    " % m
            rows.append({"arm": name, "H": H, "fano_mean": m, "fano_sem": s, "n": len(v),
                         "mean_gap": float(np.mean(mg)), "gap_cv2": float(np.mean(cv2)), "note": note})
        for L in L_list:
            v = np.array(svr[L], float); m = float(np.nanmean(v)); s = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            cells += "  %.3f +- %.3f" % (m, s) if len(v) > 1 else "  %.3f (1 seq) " % m
            rows.append({"arm": name, "H": -L, "fano_mean": m, "fano_sem": s, "n": len(v),
                         "mean_gap": float(np.mean(mg)), "gap_cv2": float(np.mean(cv2)), "note": "span-variance ratio at L=%d (H<0 marks gap coordinates)" % L})
        lines.append("  %-14s" % name + cells + "  %.2f  %.3f  %s" % (float(np.mean(mg)), float(np.mean(cv2)), note))
    lines.append("  (S1 baseline = CV^2 of the gap multiset = %.3f; Hardy-Littlewood full = %s)" % (
        gen.gap_cv2(np.diff(ctx["inner"])), ", ".join("H=%d %.3f" % (H, hl_full[H]) for H in H_list)))
    if cert_range and cert_range > N:
        lo, hi = M, M + int(cert_range)
        big = gen.segmented_primes(lo, hi)
        cells = "".join("  %.3f            " % gen.count_fano(big, lo, hi, H, stride=H) for H in H_list)
        cells += "".join("  %-14s" % ("%.3f" % gen.span_variance_ratio(np.diff(big).astype(float), L)) for L in L_list)
        lines.append("  %-14s" % ("True [M,M+%.0e)" % cert_range) + cells + "  %.2f  %.3f  disjoint windows, PNT-detrended (shell statistic)" % (
            float(np.diff(big).mean()), gen.gap_cv2(np.diff(big))))
        for H in H_list:
            rows.append({"arm": "True_range", "H": H, "fano_mean": gen.count_fano(big, lo, hi, H, stride=H), "fano_sem": float("nan"),
                         "n": 1, "mean_gap": float(np.diff(big).mean()), "gap_cv2": gen.gap_cv2(np.diff(big)), "note": "range %d" % cert_range})
    text = "\n".join(lines)
    if out_path:
        import csv
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm", "H", "fano_mean", "fano_sem", "n", "mean_gap", "gap_cv2", "note"])
            w.writeheader(); w.writerows(rows)
    return text


# =============================================================================
# Summary statistics
# =============================================================================
def _welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float(p), float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def summarize(csv_path, z_list):
    done = load_done(csv_path)
    if not done:
        return "no results yet"
    by = {}
    for (arm, run), row in done.items():
        by.setdefault(arm, {})[run] = (float(row["spearman_rho"]), float(row["dR"]), float(row["gap_cv2"]), float(row["frac_beads_at_kmax"]))
    order = ["True", "S1", "S7"] + [f"S7G_z{z}" for z in z_list] + [f"S7G_S1_z{z}" for z in z_list]
    order = [a for a in order if a in by] + [a for a in by if a not in order]
    L = ["Summary (Spearman rho: mean +- s.e.m.; Welch two-sided p and Cohen d; f_cap = fraction of beads at k_max)", ""]
    L.append("  %-14s %3s  %-18s %-9s  %-6s %-6s | %-18s | %-18s" % ("arm", "n", "rho", "dR", "CV^2", "f_cap", "vs True: p, d", "vs S1: p, d"))
    rho = {a: np.array([v[0] for v in by[a].values()]) for a in by}
    for a in order:
        v = np.array(list(by[a].values()))
        r = v[:, 0]; sem = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
        pt, dt = _welch(r, rho["True"]) if "True" in rho and a != "True" else (float("nan"), float("nan"))
        ps, ds = _welch(r, rho["S1"]) if "S1" in rho and a != "S1" else (float("nan"), float("nan"))
        L.append("  %-14s %3d  %+.4f +- %.4f  %+8.2f  %.3f  %.3f | p=%.2e d=%+.2f | p=%.2e d=%+.2f" % (
            a, len(r), r.mean(), sem, v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean(), pt, dt, ps, ds))
    if len(rho) > 2:
        F, p_anova = stats.f_oneway(*[rho[a] for a in order])
        L.append("")
        L.append("  omnibus across all %d arms: ANOVA F=%.2f p=%.3g   (if this is not small, no arm differs and"
                 " the per-arm p values below are not evidence of anything)" % (len(order), F, p_anova))
    if "True" in rho and "S1" in rho:
        L.append("")
        d_true = rho["True"].mean() - rho["S1"].mean()
        p_true = _welch(rho["True"], rho["S1"])[0]
        sd = float(np.mean([rho[a].std(ddof=1) for a in order]))
        mdd = 0.63 * sd * np.sqrt(2 / 2)   # d=0.63 is the 80%-power effect size at n=40+40
        n_true = len(rho["True"])
        mdd = 0.63 * sd * np.sqrt(40 / max(n_true, 1))
        L.append("  ordering effect  Delta_true = rho_True - rho_S1 = %+.4f  (Welch p=%.3g)" % (d_true, p_true))
        L.append("  trajectory SD = %.4f; smallest Delta this ensemble can detect at 80%% power = %.4f;"
                 " n/arm needed for the observed Delta = %s" % (
                     sd, mdd, "%.0f" % (2 * (1.96 + 0.84) ** 2 * sd ** 2 / d_true ** 2) if abs(d_true) > 1e-9 else "inf"))
        if "S7" in rho:
            den = rho["S1"].mean() - rho["True"].mean()
            f7 = (rho["S1"].mean() - rho["S7"].mean()) / den if abs(den) > 1e-9 else float("nan")
            if p_true < 0.05:
                L.append("  S7 retention  f_S7 = (rho_S1 - rho_S7)/(rho_S1 - rho_True) = %.2f"
                         "   (1 = two-point reader, 0 = shuffle-like)" % f7)
            else:
                L.append("  S7 retention  f_S7 = %.2f  NOT INTERPRETABLE: its denominator (Delta_true) is"
                         " consistent with zero at p=%.2f, so f_S7 is a ratio of noise -- ignore it until"
                         " Delta_true is established." % (f7, p_true))
        for z in z_list:
            a = f"S7G_z{z}"; b = f"S7G_S1_z{z}"
            if a in rho:
                line = "  z=%-4d  rho_S7G - rho_S1 = %+.4f" % (z, rho[a].mean() - rho["S1"].mean())
                if b in rho:
                    runs = sorted(set(by[a]) & set(by[b]))
                    d = np.array([by[a][k][0] - by[b][k][0] for k in runs])
                    tp = stats.ttest_1samp(d, 0.0).pvalue if len(d) > 1 else float("nan")
                    line += "   paired ordering effect inside S7-G: rho_S7G - rho_S7G_S1 = %+.4f (n=%d, p=%.2e)" % (d.mean(), len(d), tp)
                L.append(line)
        L.append("  (S7-G changes the gap multiset as well as its order; compare the paired column with Delta_true,")
        L.append("   and the CV^2 / f_cap columns for the composition shift.  Delta rho(z) stalling in z locates the")
        L.append("   sieve depth the fold reads.)")
    return "\n".join(L)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=float, default=1e8); ap.add_argument("--N", type=int, default=10_000)
    ap.add_argument("--steps", type=int, default=5_000); ap.add_argument("--nens", type=int, default=40)
    ap.add_argument("--z-list", type=str, default="5,11,31,100")
    ap.add_argument("--pad", type=int, default=2_000, help="extension of the sieved range on each side (integers)")
    ap.add_argument("--detrend", type=str, default="pnt", choices=["pnt", "none"], help="S7 detrending (inert at M~1e8)")
    ap.add_argument("--alpha-rescale", action="store_true", help="alpha = 0.15 ln(2e4)/ln(M+N/2)")
    ap.add_argument("--s7g-shuffle-control", action="store_true", help="add the run-paired shuffles S7G_z*_S1")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel trajectories; each worker is a full Python process (~0.4-0.8 GB with Anaconda), 3-4 is a good laptop setting")
    ap.add_argument("--maxrun", type=int, default=10 ** 9)
    ap.add_argument("--outdir", type=str, default="paper3_out")
    ap.add_argument("--certificate-range", type=float, default=1e7, help="[M, M+R) for the shell statistic of the true primes (0 = skip)")
    ap.add_argument("--certificate-only", action="store_true"); ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="N=2000, 100 steps, n=2, z=5,31")
    args = ap.parse_args()

    if args.smoke:
        args.N, args.steps, args.nens, args.z_list = 2_000, 100, 2, "5,31"
        args.certificate_range = min(args.certificate_range, 2e5); args.outdir = args.outdir + "_smoke"
    M, N = int(args.M), int(args.N)
    z_list = [int(v) for v in args.z_list.split(",") if v.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "results.csv")
    detrend = None if args.detrend == "none" else "pnt"
    alpha = ALPHA * ALPHA_REF_LNX / math.log(M + N / 2) if args.alpha_rescale else ALPHA

    if args.summarize_only:
        print(summarize(csv_path, z_list)); return

    ctx = build_context(M, N, args.pad)
    arms = [("True", None), ("S1", None), ("S7", None)] + [("S7G", z) for z in z_list]
    if args.s7g_shuffle_control:
        arms += [("S7G_S1", z) for z in z_list]
    print("window [%d, %d): %d primes inside, bracket %d .. %d, mean gap %.2f, CV^2 %.3f" % (
        M, M + N, len(ctx["inner"]), ctx["left"], ctx["right"], float(np.diff(ctx["inner"]).mean()), gen.gap_cv2(np.diff(ctx["inner"]))))
    print("gradient across the window: Delta ln i / ln i = %.1e;  alpha = %.4f (saturation at g >= %.0f);  steps = %d (%.3f/bead);  n = %d/arm" % (
        N / (M * math.log(M)), alpha, math.log(K_MAX / K_HINGE) / alpha, args.steps, args.steps / N, args.nens))
    print("arms:", ", ".join(arm_name(k, z) for k, z in arms))
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({k: (v if not isinstance(v, float) or math.isfinite(v) else str(v)) for k, v in vars(args).items()} | {"alpha_eff": alpha}, f, indent=1)

    print("\n" + certificate(ctx, arms, args.nens, z_list, detrend, cert_range=int(args.certificate_range),
                              out_path=os.path.join(args.outdir, "certificate.csv")) + "\n")
    if args.certificate_only:
        return

    done = load_done(csv_path)
    jobs = [{"ctx": ctx, "kind": k, "z": z, "run": run, "z_list": z_list, "detrend": detrend,
             "alpha": alpha, "steps": args.steps}
            for k, z in arms for run in range(args.nens) if (arm_name(k, z), run) not in done]
    total = len(arms) * args.nens
    print("trajectories: %d done, %d pending, %d total" % (len(done), len(jobs), total))
    jobs = jobs[:args.maxrun]
    t_start = time.time(); n_exec = 0

    def _tick(row):
        nonlocal n_exec
        n_exec += 1; append_row(csv_path, row)
        el = time.time() - t_start
        print("[%s run %02d] rho=%+.4f dR=%+7.2f  n_anchor=%d <g>=%.1f f_cap=%.3f  %5.1fs  (%d/%d, elapsed %.0fs, eta %.0fs)%s" % (
            row["arm"], row["run"], row["spearman_rho"], row["dR"], row["n_anchor_inside"], row["mean_gap"],
            row["frac_beads_at_kmax"], row["seconds"], n_exec, len(jobs), el, el / n_exec * (len(jobs) - n_exec),
            ("  ! " + row["warning"]) if row["warning"] else ""), flush=True)

    failures = []

    def _fail(job, exc):
        failures.append((arm_name(job["kind"], job["z"]), job["run"]))
        print("[%s run %02d] FAILED: %r   (row not written; rerun the same command to retry)" % (
            failures[-1][0], job["run"], exc), flush=True)

    if args.workers <= 1:
        for job in jobs:
            try:
                _tick(run_one_job(job))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                _fail(job, e)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one_job, job): job for job in jobs}
            for fut in as_completed(futs):
                try:
                    _tick(fut.result())
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    _fail(futs[fut], e)
    if failures:
        print("\n%d trajectories failed and were kept for retry (rerun to resume)." % len(failures))
        print("If failures are MemoryError/BrokenProcessPool, lower --workers (each worker is a full")
        print("Python process, ~0.4-0.8 GB with Anaconda; 3-4 workers is a good laptop setting).")
    text = summarize(csv_path, z_list)
    print("\n" + text)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    if len(load_done(csv_path)) < total:
        print("\n(%d trajectories still pending; rerun to continue)" % (total - len(load_done(csv_path))))


if __name__ == "__main__":
    main()
