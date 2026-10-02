# -*- coding: utf-8 -*-
"""
debye_gauss.py -- scattering function of the exact ideal PDTP chain and helpers.

S(q) = (1/N) sum_ij exp(-q^2 <R_ij^2>/6), with the exact thermal pair distances <R_ij^2> of
the ideal chain carrying the quenched bending field (core.ideal_R2_matrix, O(N^2)).
"""
from __future__ import annotations
import numpy as np
import core

Q_DEFAULT = np.geomspace(0.03, 3.0, 40)


def kappa_from_anchors(anchors, N, alpha):
    g, ia = core.gap_field(np.asarray(anchors), N)
    return core.kappa_pdtp(g, ia, alpha=alpha)


def S_of_q(kappa, q=Q_DEFAULT, beta=1.0):
    return core.debye_gaussian_closure(core.ideal_R2_matrix(kappa, beta), q)


def rho_ideal(kappa, beta=1.0):
    """Positional readout: Spearman(kappa_i, <|r_i - r_cm|^2>) of the exact ideal chain."""
    return core.spearman(kappa, core.ideal_profile(kappa, beta)["P"])


def relative_deviation(S_arm, S_ref_mean, S_ref_sd):
    d = np.asarray(S_arm) - S_ref_mean
    return d / S_ref_mean, d / (S_ref_sd + 1e-12)
