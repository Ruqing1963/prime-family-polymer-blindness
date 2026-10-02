# -*- coding: utf-8 -*-
"""
s7_generators.py -- surrogate generators for Paper 3 (the M-window design).

Conventions (integer coordinates, as in Papers 1-2)
--------------------------------------------------
An *anchor* is a prime (True arm) or a surrogate prime (S1, S7, S7-G).  The polymer
reads the gap field painted onto the composite beads between consecutive anchors
(PDTP paper, Sec. 2.1).  Every generator returns anchor positions as absolute
integers; the driver paints the field.  For the prime-indicator field the 6N
dead-residue set is dead(q) = {0} for every prime q, i.e. "n is killed by q iff
q | n"; the dead sets of any constellation follow from this member by member.

Generators
----------
generate_s1(anchors, rng)
    Paper-1 shuffle S1: random permutation of the interior gaps.  First and last
    anchor fixed, exact gap multiset.
generate_s7_iaaft(anchors, rng, detrend="pnt", ...)
    S7 (two-point only): exact gap multiset AND the power spectrum of the
    detrended gap sequence, by the Schreiber-Schmitz IAAFT algorithm; the
    surrogate is rank-mapped back onto the trend exactly as S6' does, so it is
    "S6' with correlated keys" -- the +1-ingredient arm of the Letter's hierarchy.
generate_s7_g(lo, hi, z, rng, count_mode="fixed", target_count=None)
    S7-G: Granville's refinement of the Cramer model at sieve depth z.  The dead
    residues of every prime q <= z are enforced deterministically (n survives iff
    gcd(n, P(z)) = 1); survivors become anchors at random.  Its k-point functions
    equal the Hardy-Littlewood main term truncated at z, to every order.

Certificate helpers (two-point diagnostics, Part-29 style)
-------------------------------------------------------
count_fano(anchors, lo, hi, H, ...)      Fano factor of anchor counts in windows of H integers
span_variance_ratio(gaps, L)             Var(sum of L consecutive gaps) / (L Var g)
hl_prime_count_fano(H, X, K=None)        Hardy-Littlewood prediction for the prime field
                                         (Part-29 identity with the Part-14 pair series),
                                         optionally truncated at prime cutoff K (= z)
gap_cv2(gaps)                            CV^2 of the gap multiset = count Fano of an S1 shuffle

Only numpy is required.
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "small_primes", "segmented_primes", "split_window", "gaps_from_anchors",
    "generate_s1", "iaaft", "generate_s7_iaaft", "theta_z", "generate_s7_g",
    "count_fano", "span_variance_ratio", "gap_cv2", "hl_pair_series", "hl_prime_count_fano",
    "autocorr",
]

# ----------------------------------------------------------------------------- primes
def small_primes(n: int) -> np.ndarray:
    """All primes <= n (simple sieve)."""
    n = int(n)
    if n < 2:
        return np.zeros(0, dtype=np.int64)
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0].astype(np.int64)


def segmented_primes(lo: int, hi: int) -> np.ndarray:
    """All primes p with lo <= p < hi (segmented sieve; fine up to hi ~ 1e12 for short segments)."""
    lo, hi = int(lo), int(hi)
    if hi <= 2 or hi <= lo:
        return np.zeros(0, dtype=np.int64)
    base = small_primes(int(hi ** 0.5) + 1)
    mask = np.ones(hi - lo, dtype=bool)
    for p in base:
        p = int(p)
        start = max(p * p, ((lo + p - 1) // p) * p)
        if start < hi:
            mask[start - lo::p] = False
    for v in (0, 1):                      # 0 and 1 are not prime
        if lo <= v < hi:
            mask[v - lo] = False
    return np.nonzero(mask)[0].astype(np.int64) + lo


def split_window(anchors: np.ndarray, lo: int, hi: int):
    """Split a sorted anchor array (covering more than [lo, hi)) into
    (left_bound, interior, right_bound): the last anchor < lo, the anchors in [lo, hi),
    and the first anchor >= hi.  Raises if a bound is missing."""
    a = np.asarray(anchors, dtype=np.int64)
    left = a[a < lo]; inner = a[(a >= lo) & (a < hi)]; right = a[a >= hi]
    if len(left) == 0 or len(right) == 0:
        raise ValueError("anchor set does not bracket the window; enlarge the pad")
    return int(left[-1]), inner, int(right[0])


def gaps_from_anchors(anchors: np.ndarray) -> np.ndarray:
    return np.diff(np.asarray(anchors, dtype=np.int64))


# ----------------------------------------------------------------------------- S1
def generate_s1(anchors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paper-1 S1: permute the interior gaps; first anchor (hence last) fixed."""
    a = np.asarray(anchors, dtype=np.int64)
    gaps = np.diff(a).copy(); rng.shuffle(gaps)
    return a[0] + np.concatenate(([0], np.cumsum(gaps))).astype(np.int64)


# ----------------------------------------------------------------------------- S7 (IAAFT)
def iaaft(x: np.ndarray, rng: np.random.Generator, max_iter: int = 2000):
    """Iterative amplitude-adjusted Fourier-transform surrogate (Schreiber & Schmitz 1996).

    Returns (y, info).  y has exactly the multiset of x (last step is the amplitude
    adjustment) and, to convergence accuracy, the same power spectrum as x.
    info = dict(iterations, spectral_error) with
    spectral_error = ||amp(y) - amp(x)||_2 / ||amp(x)||_2 over the rfft amplitudes.
    """
    x = np.asarray(x, dtype=float); n = len(x)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    prev_rank = None; it = 0
    for it in range(1, max_iter + 1):
        Y = np.fft.rfft(y)
        y_spec = np.fft.irfft(target_amp * np.exp(1j * np.angle(Y)), n=n)   # impose spectrum
        rank = np.argsort(np.argsort(y_spec, kind="stable"), kind="stable")
        y = sorted_x[rank]                                                   # impose multiset
        if prev_rank is not None and np.array_equal(rank, prev_rank):
            break
        prev_rank = rank
    amp = np.abs(np.fft.rfft(y))
    err = float(np.linalg.norm(amp - target_amp) / (np.linalg.norm(target_amp) + 1e-300))
    return y, {"iterations": it, "spectral_error": err}


def _pnt_local_mean_gap(left_anchor_positions: np.ndarray) -> np.ndarray:
    """Expected local gap ln(p) at the left prime of each gap (prime number theorem)."""
    return np.maximum(np.log(np.asarray(left_anchor_positions, dtype=float)), 1.0)


def generate_s7_iaaft(anchors: np.ndarray, rng: np.random.Generator,
                      detrend: str | None = "pnt", max_iter: int = 2000):
    """S7: exact interior-gap multiset + power spectrum of the detrended gap sequence.

    Steps: (1) detrend g_k -> x_k = g_k / m_k with m_k = ln p_k (PNT local mean gap;
    at M ~ 1e8 this is constant to 1e-4 and the step is inert), (2) IAAFT on x,
    (3) keys kappa_k = m_k * x~_k, (4) rank-map the true gap multiset onto the keys
    (the r-th smallest gap goes to the slot with the r-th smallest key), exactly the
    S6' construction with correlated keys.  First and last anchor fixed.
    Returns (new_anchors, info)."""
    a = np.asarray(anchors, dtype=np.int64)
    gaps = np.diff(a).astype(float)
    if len(gaps) < 8:
        raise ValueError("too few gaps for a spectral surrogate")
    if detrend == "pnt":
        m = _pnt_local_mean_gap(a[:-1])
    elif detrend is None or detrend == "none":
        m = np.ones_like(gaps)
    else:
        raise ValueError("detrend must be 'pnt' or None")
    x = gaps / m
    xs, info = iaaft(x, rng, max_iter=max_iter)
    keys = m * xs
    order = np.argsort(keys, kind="stable")
    new_gaps = np.empty_like(gaps); new_gaps[order] = np.sort(gaps)
    new_anchors = a[0] + np.concatenate(([0], np.cumsum(new_gaps))).astype(np.int64)
    assert new_anchors[-1] == a[-1], "gap multiset not preserved"
    info = dict(info)
    info["acf_true_1_3"] = autocorr(x, 3).tolist()
    info["acf_s7_1_3"] = autocorr(xs, 3).tolist()
    return new_anchors, info


# ----------------------------------------------------------------------------- S7-G (Granville-Cramer)
def theta_z(z: int) -> float:
    """theta_z = prod_{q <= z} (1 - 1/q)^{-1}: the Granville density correction."""
    return float(np.prod([1.0 / (1.0 - 1.0 / q) for q in small_primes(z)])) if z >= 2 else 1.0


def generate_s7_g(lo: int, hi: int, z: int, rng: np.random.Generator,
                  count_mode: str = "fixed", target_count: int | None = None):
    """S7-G: Granville's refinement of the Cramer model on [lo, hi) at sieve depth z.

    Deterministic part: n is a candidate iff n mod q != 0 for every prime q <= z
    (the dead residue dead(q) = {0} of every q <= z is enforced exactly).
    Random part:
      count_mode="bernoulli": each candidate becomes an anchor independently with
          probability theta_z / ln n  (the literal Granville model; density 1/ln n).
      count_mode="fixed":     exactly target_count anchors are drawn uniformly among
          the candidates (default).  Conditions on the anchor count, as S1 and S7
          condition on the gap multiset; use the true prime count of [lo, hi).
    Returns (anchors, info).  The k-point functions of the bernoulli model equal the
    Hardy-Littlewood main term truncated at z to every order; the fixed-count variant
    differs only by the global count conditioning.
    """
    lo, hi, z = int(lo), int(hi), int(z)
    qs = small_primes(z)
    n = np.arange(lo, hi, dtype=np.int64)
    cand = np.ones(hi - lo, dtype=bool)
    for q in qs:
        q = int(q)
        start = ((lo + q - 1) // q) * q
        if start < hi:
            cand[start - lo::q] = False
    th = theta_z(z)
    if count_mode == "bernoulli":
        p = np.minimum(1.0, th / np.log(np.maximum(n, 3)))
        keep = cand & (rng.random(hi - lo) < p)
        anchors = n[keep]
    elif count_mode == "fixed":
        idx = np.nonzero(cand)[0]
        if target_count is None:
            target_count = int(round(float(np.sum(1.0 / np.log(np.maximum(n, 3))))))
        if target_count > len(idx):
            raise ValueError("target_count exceeds the number of candidates")
        chosen = np.sort(rng.choice(idx, size=int(target_count), replace=False))
        anchors = n[chosen]
    else:
        raise ValueError("count_mode must be 'fixed' or 'bernoulli'")
    info = {"z": z, "theta_z": th, "n_candidates": int(cand.sum()),
            "candidate_density": float(cand.mean()), "n_anchors": int(len(anchors)),
            "survival_prob": float(th / np.log(0.5 * (lo + hi)))}
    return anchors.astype(np.int64), info


# ----------------------------------------------------------------------------- certificate helpers
def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag."""
    x = np.asarray(x, dtype=float); x = x - x.mean(); v = float(np.dot(x, x))
    return np.array([np.dot(x[:-k], x[k:]) / v for k in range(1, max_lag + 1)])


def gap_cv2(gaps: np.ndarray) -> float:
    """CV^2 of the gap multiset: the count Fano factor of an S1-type shuffle (renewal limit)."""
    g = np.asarray(gaps, dtype=float)
    return float(g.var() / g.mean() ** 2)


def count_fano(anchors: np.ndarray, lo: int, hi: int, H: int,
               stride: int | None = None, detrend: bool = True) -> float:
    """Fano factor Var/mean of anchor counts in windows of H integers tiling [lo, hi)
    (stride = H: disjoint windows; stride < H: sliding windows, a less noisy but
    correlated estimate).  detrend=True subtracts the PNT expectation sum 1/ln x."""
    a = np.asarray(anchors, dtype=np.int64); a = a[(a >= lo) & (a < hi)]
    stride = int(H if stride is None else stride)
    starts = np.arange(lo, hi - H + 1, stride)
    if len(starts) < 2:
        return float("nan")
    cum = np.searchsorted(a, starts + H) - np.searchsorted(a, starts)
    c = cum.astype(float)
    if detrend:
        xs = np.arange(lo, hi, dtype=float); w = np.concatenate(([0.0], np.cumsum(1.0 / np.log(np.maximum(xs, 3)))))
        e = w[starts - lo + H] - w[starts - lo]
        var = float(np.mean((c - e) ** 2)) * len(c) / (len(c) - 1)
    else:
        var = float(c.var(ddof=1))
    return var / float(c.mean())


def span_variance_ratio(gaps: np.ndarray, L: int) -> float:
    """Var(sum of L consecutive gaps) / (L Var g), sliding windows.
    = 1 + 2 sum_{t<L} (1 - t/L) C(t) for a stationary sequence: 1 for i.i.d. gaps (S1),
    < 1 when the sequence is hyperuniform (rigid) at that scale."""
    g = np.asarray(gaps, dtype=float)
    if len(g) <= L + 1:
        return float("nan")
    cs = np.concatenate(([0.0], np.cumsum(g)))
    spans = cs[L:] - cs[:-L]
    return float(spans.var(ddof=1) / (L * g.var(ddof=1)))


def _distinct_prime_factors(h: int, primes: np.ndarray):
    out = []; m = int(h)
    for p in primes:
        p = int(p)
        if p * p > m:
            break
        if m % p == 0:
            out.append(p)
            while m % p == 0:
                m //= p
    if m > 1:
        out.append(m)
    return out


def hl_pair_series(h: int, primes: np.ndarray, K: int | None = None) -> float:
    """Hardy-Littlewood pair singular series S(h) = 2 C2 prod_{p|h, p>2} (p-1)/(p-2) for even h
    (0 for odd h), with the Euler product truncated at p <= K if K is given (K = sieve depth z)."""
    if h % 2:
        return 0.0
    ps = primes if K is None else primes[primes <= K]
    c = 2.0 * float(np.prod([1.0 - 1.0 / (p - 1) ** 2 for p in ps if p > 2]))
    for p in _distinct_prime_factors(h, primes):
        if p > 2 and (K is None or p <= K):
            c *= (p - 1) / (p - 2)
    return c


def hl_prime_count_fano(H: int, X: float, K: int | None = None, prime_table: np.ndarray | None = None) -> float:
    """Part-29 variance identity for the prime-indicator field at local density rho = 1/ln X:
        F(H) = (1 - rho) + (2 rho / H) sum_{h<H} (H - h) (S(h) - 1),
    S = Part-14 pair series (K = None: full product; K = z: truncated as in S7-G at depth z).
    Valid for H << X (stationary approximation)."""
    H = int(H)
    primes = small_primes(max(H, 1000) if prime_table is None else 0) if prime_table is None else prime_table
    rho = 1.0 / np.log(X)
    s = sum((H - h) * (hl_pair_series(h, primes, K) - 1.0) for h in range(1, H))
    return float((1.0 - rho) + 2.0 * rho / H * s)


# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # sieve check
    lo, hi = 10 ** 8 - 2000, 10 ** 8 + 12000
    ps = segmented_primes(lo, hi)
    brute = [n for n in range(lo, lo + 3000) if n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))]
    assert list(ps[ps < lo + 3000]) == brute, "segmented sieve mismatch"
    left, inner, right = split_window(ps, 10 ** 8, 10 ** 8 + 10 ** 4)
    print("primes in [1e8, 1e8+1e4):", len(inner), " bounds:", left, right)
    gaps = gaps_from_anchors(inner)
    print("gap multiset: n=%d <g>=%.2f CV^2=%.3f  acf(1..3)=%s" % (len(gaps), gaps.mean(), gap_cv2(gaps),
          np.round(autocorr(gaps, 3), 3)))
    # S1 / S7
    s1 = generate_s1(inner, rng); assert s1[-1] == inner[-1] and np.array_equal(np.sort(np.diff(s1)), np.sort(gaps))
    s7, info = generate_s7_iaaft(inner, rng)
    assert np.array_equal(np.sort(np.diff(s7)), np.sort(gaps))
    print("S7 IAAFT: iterations=%d spectral_error=%.3e  acf true=%s  acf S7=%s" % (
        info["iterations"], info["spectral_error"], np.round(info["acf_true_1_3"], 3), np.round(info["acf_s7_1_3"], 3)))
    for L in (2, 5, 10, 20, 50):
        print("  L=%3d span-variance ratio: true %.3f  S7 %.3f  S1 %.3f" % (
            L, span_variance_ratio(gaps, L), span_variance_ratio(np.diff(s7), L), span_variance_ratio(np.diff(s1), L)))
    # S7-G
    for z in (5, 11, 31, 100):
        g, gi = generate_s7_g(lo, hi, z, rng, count_mode="fixed", target_count=len(ps))
        gg = np.diff(g[(g >= 10 ** 8) & (g < 10 ** 8 + 10 ** 4)])
        print("S7-G z=%3d: theta=%.3f cand.density=%.3f survival p=%.3f  n_anchor=%d  <g>=%.2f CV^2=%.3f" % (
            z, gi["theta_z"], gi["candidate_density"], gi["survival_prob"], gi["n_anchors"], gg.mean(), gap_cv2(gg)))
    # HL prediction at X=1e8
    for H in (100, 1000):
        print("HL count Fano at X=1e8, H=%d: full %.3f | z=5 %.3f | z=11 %.3f | z=31 %.3f | z=100 %.3f" % (
            H, hl_prime_count_fano(H, 1e8), *[hl_prime_count_fano(H, 1e8, K=z) for z in (5, 11, 31, 100)]))
