# -*- coding: utf-8 -*-
"""
sieve_models.py -- prime families, Granville-Cramer sieve models S7-G(z), shuffles, Poisson
baselines, and the mod-q jitter surrogates (kill / keep / variance-matched keep).

All anchor sets are sorted integer arrays of positions in [0, N).  The polymer reads the gap
field painted between consecutive anchors (core.gap_field).

Families (anchors below N):
    primes   all primes
    twin     p with p + 2 prime            (p > 3: p = 2 mod 3, gaps in 6Z)
    SG       p with 2p + 1 prime           (p > 3: p = 2 mod 3, gaps in 6Z)
    Landau   primes of the form n^2 + 1    (odd members: = 1 mod 4, gaps in 4Z)

Sieve model S7-G(z): the integers that pass the family's OWN sieve conditions at depth z
(primes: n; twin: n and n+2; SG: n and 2n+1; Landau: n^2+1), each value coprime to
P(z) = prod_{p <= z} p or itself a prime <= z.  z = 1 means no sieve (for Landau: the
polynomial support n^2+1 only).  Draws are BINNED: bins hold Q consecutive family members and
each bin receives the family's own count, so the anchor count and the smooth density envelope
(PNT drift, quadratic spacing) are held fixed at resolution Q.
"""
from __future__ import annotations
import numpy as np

FAMILIES = ("primes", "twin", "SG", "Landau")


def sieve(n: int) -> np.ndarray:
    ip = np.ones(n + 1, dtype=bool); ip[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if ip[i]:
            ip[i * i::i] = False
    return ip


def family(name: str, N: int, ip: np.ndarray | None = None) -> np.ndarray:
    """Members of the family below N (ip must cover 2N+2 for SG)."""
    if ip is None:
        ip = sieve(2 * N + 4)
    P = np.where(ip[:N])[0]
    if name == "primes":
        return P
    if name == "twin":
        return P[ip[P + 2]]
    if name == "SG":
        return P[ip[2 * P + 1]]
    if name == "Landau":
        n = np.arange(1, int(np.sqrt(N)) + 2); v = n * n + 1; v = v[v < N]
        return v[ip[v]]
    raise ValueError(name)


def ok_mask(M: int, z: int, ip: np.ndarray) -> np.ndarray:
    """ok[v] = v coprime to P(z), or v itself a prime <= z  (v < M)."""
    ok = np.ones(M, bool)
    for p in [p for p in range(2, z + 1) if ip[p]]:
        ok[::p] = False
        ok[p] = True
    ok[:2] = False
    return ok


def candidates(name: str, N: int, z: int, ip: np.ndarray) -> np.ndarray:
    """Sorted positions < N passing the family's sieve conditions at depth z."""
    if name == "Landau":
        n = np.arange(1, int(np.sqrt(N)) + 2); v = n * n + 1; v = v[v < N]
        if z <= 1:
            return v
        ok = ok_mask(int(v.max()) + 1, z, ip)
        return v[ok[v]]
    x = np.arange(2, N)
    if z <= 1:
        return x
    ok = ok_mask(2 * N + 3, z, ip)
    if name == "primes":
        return x[ok[x]]
    if name == "twin":
        return x[ok[x] & ok[x + 2]]
    if name == "SG":
        return x[ok[x] & ok[2 * x + 1]]
    raise ValueError(name)


def bins_from_family(a: np.ndarray, N: int, Q: int):
    """Bin edges holding Q consecutive members each, and the member count per bin."""
    cut = [(int(a[j - 1]) + int(a[j])) // 2 + 1 for j in range(Q, len(a), Q)]
    edges = np.array([0] + cut + [N])
    return edges, np.histogram(a, bins=edges)[0]


def binned_draw(cand: np.ndarray, edges: np.ndarray, cnt: np.ndarray, rng) -> np.ndarray:
    """Uniform draw without replacement inside each bin, with the prescribed count per bin."""
    idx = np.searchsorted(cand, edges)
    out = []
    for b in range(len(cnt)):
        pool = cand[idx[b]:idx[b + 1]]
        k = min(int(cnt[b]), len(pool))
        if k:
            out.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(out))


def s7g(name: str, a: np.ndarray, N: int, z: int, rng, ip: np.ndarray, Q: int = 20) -> np.ndarray:
    edges, cnt = bins_from_family(a, N, Q)
    return binned_draw(candidates(name, N, z, ip), edges, cnt, rng)


def poisson(a: np.ndarray, N: int, rng, Q: int = 20) -> np.ndarray:
    """Binned uniform draw from all integers: envelope and count kept, no support, no sieve."""
    edges, cnt = bins_from_family(a, N, Q)
    return binned_draw(np.arange(2, N), edges, cnt, rng)


def shuffle(a: np.ndarray, rng) -> np.ndarray:
    """S1: permute the gaps, first anchor fixed (exact gap multiset)."""
    return a[0] + np.concatenate(([0], np.cumsum(rng.permutation(np.diff(a)))))


# ------------------------------------------------------------------ mod-q jitter surrogates
def jitter(a: np.ndarray, q: int, rng, arm: str) -> np.ndarray:
    """Jittered anchor set, re-sorted.
    arm = 'kill'   : shift ~ U{0..q-1} - floor((q-1)/2)   (nulls every divisor channel of q)
          'keep'   : shift = q * U{0,1}                    (v1 keep arm; preserves divisor channels)
          'keep_vm': shift = q * b, P(b=+-1) = (q^2-1)/(24 q^2)   (variance-matched keep arm)
    """
    n = len(a)
    if arm == "kill":
        eps = rng.integers(0, q, n) - (q - 1) // 2
    elif arm == "keep":
        eps = q * rng.integers(0, 2, n)
    elif arm == "keep_vm":
        p = (q * q - 1) / (24 * q * q); u = rng.random(n)
        eps = q * ((u < p).astype(int) - ((u >= p) & (u < 2 * p)).astype(int))
    else:
        raise ValueError(arm)
    # The two end anchors stay fixed (as in S1) and every jittered anchor stays between them.
    # Without this, a negative first anchor corrupts the gap field (core.gap_field paints
    # g[:a[0]] with a negative slice index).
    eps[0] = 0; eps[-1] = 0
    return np.sort(np.clip(a + eps, a[0], a[-1]))


def bragg_power(positions: np.ndarray, q: int) -> float:
    """Mean squared Bragg amplitude of the point set at a/q over a coprime to q."""
    x = np.asarray(positions, float)
    a_vals = [a for a in range(1, q) if np.gcd(a, q) == 1]
    return float(np.mean([np.abs(np.exp(2j * np.pi * x * a / q).mean()) ** 2 for a in a_vals]))
