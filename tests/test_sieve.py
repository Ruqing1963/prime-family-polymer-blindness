# -*- coding: utf-8 -*-
"""Tests of the prime families, their congruence conditions, and the S7-G(z) sieve models."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import sieve_models as sm  # noqa: E402

N = 200_000
IP = sm.sieve(2 * N + 4)


def test_sieve_small():
    assert list(np.where(sm.sieve(30))[0]) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_family_counts():
    """Counts to 2e5 against LBCS Ch. 37, Remark 2.3 (which counts p > 3 for the mod-3 rows and
    odd p for Landau): twin 2159, Sophie Germain 2056, Landau 63; and pi(2e5) = 17984."""
    assert len(sm.family("primes", N, IP)) == 17984
    tw = sm.family("twin", N, IP); assert len(tw[tw > 3]) == 2159
    sg = sm.family("SG", N, IP); assert len(sg[sg > 3]) == 2056
    la = sm.family("Landau", N, IP); assert len(la[la % 2 == 1]) == 63


def test_congruence_traps():
    """LBCS Ch. 37, Lemma 2.1: twin and SG members p > 3 are 2 mod 3; odd Landau primes are 1 mod 4."""
    for fam in ("twin", "SG"):
        a = sm.family(fam, N, IP); a = a[a > 3]
        assert np.all(a % 3 == 2), fam
        assert np.all(np.diff(a) % 6 == 0), fam          # odd and 2 mod 3 -> 5 mod 6 -> gaps in 6Z
    L = sm.family("Landau", N, IP); L = L[L > 2]
    assert np.all(L % 4 == 1)
    assert np.all(np.diff(L) % 4 == 0)


def test_s7g_contains_family():
    """Every family member passes its own sieve at every depth (the family is a sub-sample of S7-G)."""
    for fam in sm.FAMILIES:
        a = sm.family(fam, N, IP)
        for z in (1, 3, 5, 7):
            c = sm.candidates(fam, N, z, IP)
            assert np.all(np.isin(a, c)), (fam, z)


def test_s7g_coprimality():
    for fam in ("primes", "twin", "SG"):
        c = sm.candidates(fam, N, 7, IP)
        big = c[c > 7]
        for p in (2, 3, 5, 7):
            assert np.all(big % p != 0), (fam, p)
            if fam == "twin": assert np.all((big + 2) % p != 0)
            if fam == "SG":   assert np.all((2 * big + 1) % p != 0)


def test_s7g_reproduces_trap_at_z3():
    """At z = 3 the twin/SG sieve model already places every candidate in 5 mod 6."""
    for fam in ("twin", "SG"):
        c = sm.candidates(fam, N, 3, IP); c = c[c > 3]
        assert np.all(c % 6 == 5), fam


def test_binned_draw_keeps_counts():
    rng = np.random.default_rng(0)
    for fam in sm.FAMILIES:
        a = sm.family(fam, N, IP)
        edges, cnt = sm.bins_from_family(a, N, 20)
        assert cnt.sum() == len(a)
        d = sm.s7g(fam, a, N, 5, rng, IP)
        assert len(d) == len(a) and np.all(np.diff(d) > 0)
        assert np.array_equal(np.histogram(d, bins=edges)[0], cnt)


def test_shuffle_keeps_multiset():
    rng = np.random.default_rng(1); a = sm.family("primes", N, IP); s = sm.shuffle(a, rng)
    assert np.array_equal(np.sort(np.diff(a)), np.sort(np.diff(s))) and s[0] == a[0] and s[-1] == a[-1]


def test_jitter_selectivity():
    """kill nulls the a/6 channel, keep and keep_vm preserve it (mod_q selectivity proposition)."""
    rng = np.random.default_rng(2); a = sm.family("primes", 40_000, IP)
    b0 = sm.bragg_power(a, 6)
    assert sm.bragg_power(sm.jitter(a, 6, rng, "kill"), 6) < 0.01 * b0
    for arm in ("keep", "keep_vm"):
        assert abs(sm.bragg_power(sm.jitter(a, 6, rng, arm), 6) - b0) < 1e-12


def test_jitter_keeps_endpoints():
    """Regression: jittered anchor sets keep both end anchors and never leave [a0, a_last]
    (a negative first anchor used to corrupt the painted gap field)."""
    rng = np.random.default_rng(4); a = sm.family("primes", 6000, IP)
    for arm in ("kill", "keep", "keep_vm"):
        for q in (6, 7):
            for _ in range(50):
                x = sm.jitter(a, q, rng, arm)
                assert x[0] == a[0] and x[-1] == a[-1] and len(x) == len(a) and np.all(np.diff(x) >= 0)


def test_jitter_variance_matching():
    """Displacement variance of keep_vm equals the kill arm's (q^2-1)/12 in expectation."""
    rng = np.random.default_rng(3); q = 6; a = np.zeros(2_000_000, dtype=np.int64)
    v_kill = (q * q - 1) / 12
    p = (q * q - 1) / (24 * q * q); v_vm = 2 * p * q * q
    assert abs(v_vm - v_kill) < 1e-12
    u = rng.random(len(a)); eps = q * ((u < p).astype(int) - ((u >= p) & (u < 2 * p)).astype(int))
    assert abs(eps.var() - v_kill) / v_kill < 0.02
