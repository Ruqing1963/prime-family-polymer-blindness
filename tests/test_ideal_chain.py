# -*- coding: utf-8 -*-
"""Benchmarks of the exact ideal-chain solver (core.py, from PDTP v4.0)."""
import os, sys, math
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import core           # noqa: E402
import debye_gauss as dg  # noqa: E402


def brute_R2(kappa):
    """O(N^3) reference: <R_ij^2> = sum over bond pairs in the window of prod ell."""
    ell = core.langevin_L(np.asarray(kappa, float)); N = len(kappa); nb = N - 1
    W = np.ones((nb, nb))
    for a in range(nb):
        for b in range(a + 1, nb):
            W[a, b] = W[b, a] = np.prod(ell[a + 1:b + 1])
    R2 = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            R2[i, j] = R2[j, i] = W[i:j, i:j].sum()
    return R2


def test_langevin_function():
    x = np.array([1e-6, 0.5, 2.0, 50.0])
    ref = np.array([1e-6 / 3, 1 / math.tanh(0.5) - 2, 1 / math.tanh(2) - 0.5, 1 - 1 / 50])
    assert np.allclose(core.langevin_L(x), ref, rtol=1e-6)


def test_profile_matches_brute_force():
    rng = np.random.default_rng(0); kappa = rng.uniform(0.1, 20, 40)
    R2 = brute_R2(kappa); pr = core.ideal_profile(kappa)
    N = len(kappa); Q = R2.sum(1)
    assert np.allclose(pr["Q"], Q, rtol=1e-12)
    assert abs(pr["Rg2"] - R2.sum() / (2 * N * N)) < 1e-10
    assert abs(pr["Ree2"] - R2[0, -1]) < 1e-9
    assert np.allclose(core.ideal_R2_matrix(kappa), R2, rtol=1e-12)


def test_homogeneous_u_profile():
    """Homogeneous chain: ends peripheral, middle central; symmetric profile."""
    pr = core.ideal_profile(np.full(2001, 2.0)); P = pr["P"]
    assert np.allclose(P, P[::-1], rtol=1e-10)
    assert P[0] > P[1000] and np.argmin(P) == 1000


def test_rg_against_exact_samples():
    rng = np.random.default_rng(1); kappa = rng.uniform(0.25, 10, 300)
    confs = core.sample_ideal(kappa, n_conf=4000, seed=3)
    rg2 = np.mean([((c - c.mean(0)) ** 2).sum(1).mean() for c in confs])
    assert abs(rg2 / core.ideal_profile(kappa)["Rg2"] - 1) < 0.03


def test_spearman_matches_scipy():
    rng = np.random.default_rng(2); x = rng.integers(0, 5, 500).astype(float); y = rng.normal(size=500)
    assert abs(core.spearman(x, y) - stats.spearmanr(x, y).correlation) < 1e-12


def test_curvature_energy_identity():
    """<k(1 - cos theta)> = k (1 - L(k)) at beta = 1 (mean of the exact angle law)."""
    k = 3.0; confs = core.sample_ideal(np.full(4000, k), n_conf=1, seed=5)[0]
    t = np.diff(confs, axis=0); c = (t[1:] * t[:-1]).sum(1)
    assert abs(np.mean(k * (1 - c)) - k * (1 - core.langevin_L(np.array([k]))[0])) < 0.05


def test_debye_rigid_and_limits():
    """S(q -> 0) = N; S(q) decreasing in q."""
    kappa = np.full(200, 1.0); q = np.array([1e-4, 0.1, 1.0, 3.0])
    S = dg.S_of_q(kappa, q)
    assert abs(S[0] - 200) / 200 < 1e-3 and np.all(np.diff(S) < 0)


def test_rho_ideal_sign_for_drift():
    """A stiffness that rises along the chain makes rho_ideal positive-or-negative by the end effect;
    reversing the chain leaves rho unchanged (profile symmetry)."""
    kappa = np.linspace(0.5, 20, 3001)
    assert abs(dg.rho_ideal(kappa) - dg.rho_ideal(kappa[::-1])) < 1e-12
