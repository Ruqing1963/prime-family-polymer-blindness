#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_figures.py -- all figures of the paper from data/*.csv (writes paper/figures/*.pdf/png).
Fig. 3  composition distance to S7-G(z) and rho blindness   (p3_composition_results.csv)
Fig. 2  S(q) of jitter arms relative to the shuffle ensemble (sq_channel_results.csv)
Fig. 1  drift-free window: true chain vs ten fixed shuffles   (m_window_*.csv)"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); FIG = os.path.join(HERE, "..", "paper", "figures")
FAM = ["primes", "twin", "SG", "Landau"]
LAB = {"primes": "primes", "twin": "twin", "SG": "Sophie Germain", "Landau": r"Landau $n^2+1$"}
COL = {"primes": "#1f77b4", "twin": "#d62728", "SG": "#2ca02c", "Landau": "#9467bd"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     # embed TrueType (Type 42) fonts and use STIX for mathtext, so that Greek
                     # letters such as rho survive in every PDF viewer and in LaTeX inclusion
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def read(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig); print("written", name)


def fig1():
    rows = read("p3_composition_results.csv")
    get = lambda fam, arm, key: float(next(r[key] for r in rows if r["family"] == fam and r["arm"] == arm))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    zs = [1, 3, 5, 7]
    for fam in FAM:
        y = [max(get(fam, f"S7G_z{z}", "KS_excess"), 0.0) for z in zs]
        ax[0].plot(zs, y, "o-", color=COL[fam], label=LAB[fam], ms=4)
        ax[0].plot([8.5], [max(get(fam, "Poisson", "KS_excess"), 0.0)], "s", mfc="none", color=COL[fam], ms=4)
    ax[0].set_xticks([1, 3, 5, 7, 8.5]); ax[0].set_xticklabels(["1", "3", "5", "7", "Poisson"])
    ax[0].set_xlabel("sieve depth $z$ of the model $S_7$-G$(z)$"); ax[0].set_ylabel("composition distance (KS excess)")
    ax[0].set_title("(a) stiffness composition converges with sieve depth"); ax[0].legend(frameon=False)
    arms = ["S1", "S7G_z1", "S7G_z3", "S7G_z5", "S7G_z7", "Poisson"]
    w = 0.2
    for i, fam in enumerate(FAM):
        ax[1].bar(np.arange(len(arms)) + (i - 1.5) * w, [get(fam, a, "z_rho_id") for a in arms], w, color=COL[fam], label=LAB[fam])
    ax[1].axhline(0, color="k", lw=0.6); ax[1].axhspan(-2, 2, color="0.9", zorder=0)
    ax[1].set_xticks(range(len(arms))); ax[1].set_xticklabels(["S1", "$z{=}1$", "$z{=}3$", "$z{=}5$", "$z{=}7$", "Poisson"])
    ax[1].set_ylabel(r"$z$ of $\rho_{\mathrm{id}}$ (True $-$ arm)"); ax[1].set_title(r"(b) $\rho$ separates only the envelope-destroying shuffle")
    fig.tight_layout(); save(fig, "fig3_composition")


def fig2():
    rows = read("sq_channel_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    style = {"True": ("k", "-", 1.6), "kill6": ("#d62728", "-", 1), "keep6": ("#1f77b4", "-", 1), "keep_vm6": ("#2ca02c", "-", 1),
             "kill7": ("#d62728", "--", 1), "keep7": ("#1f77b4", "--", 1), "keep_vm7": ("#2ca02c", "--", 1)}
    for a, fam in zip(ax, ["primes", "twin"]):
        rr = [r for r in rows if r["family"] == fam]
        q = np.array(sorted({float(r["q"]) for r in rr}))
        ref = [r for r in rr if r["arm"] == "True"]; m = np.array([float(r["S1_mean"]) for r in ref]); s = np.array([float(r["S1_sd"]) for r in ref])
        a.fill_between(q, -s / m, s / m, color="0.85", label="S1 ensemble $\\pm1$ s.d.")
        for arm, (c, ls, lw) in style.items():
            y = [float(r["rel"]) for r in rr if r["arm"] == arm]
            if y: a.plot(q, y, ls, color=c, lw=lw, label=arm.replace("keep_vm", "keep*").replace("6", " $q{=}6$").replace("7", " $q{=}7$"))
        a.set_xscale("log"); a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("$q$"); a.set_ylabel(r"$[S(q)-\bar S_{\rm S1}(q)]/\bar S_{\rm S1}(q)$")
        a.set_title(f"({'a' if fam == 'primes' else 'b'}) {LAB[fam]}, $N={int(rr[0]['N'])}$")
    ax[1].legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(); save(fig, "fig2_sq_channel")


def fig3():
    main = {}
    for r in read("m_window_results.csv"):
        main.setdefault(r["arm"], []).append(float(r["spearman_rho"]))
    re = {}
    for r in read("m_window_random_effects.csv"):
        re.setdefault(int(r["seq_id"]), []).append(float(r["spearman_rho"]))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    T = np.array(main["True"]); tm, ts = T.mean(), T.std(ddof=1) / np.sqrt(len(T))
    ax.axhspan(tm - ts, tm + ts, color="0.8"); ax.axhline(tm, color="k", lw=1.2, label="true primes ($n=40$)")
    for k, v in sorted(re.items()):
        v = np.array(v); ax.errorbar(k, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="#1f77b4", ms=3.5, lw=0.8)
    mu = np.mean([np.mean(v) for v in re.values()])
    ax.axhline(mu, color="#1f77b4", ls="--", lw=1, label="fixed-shuffle class mean")
    for i, (arm, c) in enumerate([("S1", "#ff7f0e"), ("S7", "#2ca02c"), ("S7G_z5", "#9467bd"), ("S7G_S1_z5", "#8c564b")]):
        v = np.array(main[arm]); ax.errorbar(11.5 + i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt="s", color=c, ms=3.5, lw=0.8)
    ax.set_xticks(list(range(10)) + [11.5, 12.5, 13.5, 14.5])
    ax.set_xticklabels([str(k) for k in range(10)] + ["S1", "S7", "G5", "G5-S1"], rotation=90, fontsize=6)
    ax.set_xlabel("fixed shuffles $k$ ($m=5$)  |  fresh arms ($n=40$)")
    ax.set_ylabel(r"$\rho$"); ax.legend(frameon=False, loc="lower left")
    ax.set_title(r"drift-free window $[10^8, 10^8+10^4)$: $Z_{\rm seq}=+1.04$")
    fig.tight_layout(); save(fig, "fig1_m_window")


if __name__ == "__main__":
    fig1(); fig3()
    if os.path.exists(os.path.join(DATA, "sq_channel_results.csv")): fig2()
    else: print("sq_channel_results.csv not yet present; Fig. 2 skipped")
