# What a prime-programmed polymer can read

Sieve composition, congruence traps, and arithmetic blindness. Paper 3 of the
prime-driven topological polymer (PDTP) series.

> R. Chen, *What a prime-programmed polymer can read: sieve composition, congruence traps,
> and arithmetic blindness* (2026). DOI: [10.5281/zenodo.23094095](https://doi.org/10.5281/zenodo.23094095).
> Repository: https://github.com/Ruqing1963/prime-family-polymer-blindness

## The question

A self-avoiding chain whose bending stiffness follows the gaps of a set of primes registers the
**ordering** of the gaps. Papers 1–2 showed that this imprint is the smooth prime-number-theorem
envelope at the small-prime end of the chain. This paper asks whether the fold can register
anything finer. The target is **congruence structure**: twin and Sophie Germain primes are
confined to 2 mod 3, and odd Landau primes n²+1 to 1 mod 4. In localized Bost–Connes systems,
that confinement decides whether the equilibrium state is unique (*Localized Bost–Connes
Systems*, Ch. 37).

## The answer

The polymer reads arithmetic through two channels only: the positional envelope and the
sieve-depth composition of the gap multiset.

| Test | Result |
|---|---|
| **Theorem** (order blindness) | Every True-vs-shuffle contrast is blind to congruence channels locked by the gap multiset, which includes every single-class confinement. |
| Drift-free window [10⁸, 10⁸+10⁴), Langevin | Primes vs 10 fixed shuffles × 5 trajectories: Z_seq = +1.04, p = 0.33. No effect. |
| Exact positional readout, mod-q jitter arms | Kill/keep contrasts ≤ 0.004 against a shuffle spread of 0.02. |
| Exact scattering function S(q) | No channel response. The apparent shifts are jitter-induced composition changes. No jitter pair can avoid this. |
| Composition vs Granville sieve models S7-G(z) | Monotone convergence with sieve depth z. The congruence trap is the z = 3 layer. Twin primes lie *further* from their sieve model than Landau primes. A congruence condition acts in proportion to modulus / mean gap. |

The kernel-versus-generator dichotomy of the Bost–Connes setting has no polymer counterpart.

## Layout

```
paper/        main.tex, references.bib, figures/      (pdflatex + bibtex)
code/         core.py            exact ideal-chain solver (from PDTP v4.0, unmodified)
              sieve_models.py    families, S7-G(z) binned sieve models, S1, Poisson, mod-q jitter
              debye_gauss.py     S(q) in the Debye-Gauss closure, positional readout
              m_window_driver.py Langevin engine for the drift-free window (from the PDTP series)
              s7_generators.py   its surrogate generators
scripts/      run_composition_test.py   Table IV, Fig. 3       (~10 min)
              run_sq_channel_test.py    Tables II-III, Fig. 2  (~40 min; --quick for minutes)
              run_m_window_re.py        Table I, Fig. 1        (seconds from CSV; --rerun ~15 min, 12 cores)
              plot_figures.py           all figures from data/
data/         p3_composition_results.csv, rho_channel_results.csv, jitter_audit.csv,
              sq_channel_results.csv, m_window_results.csv, m_window_random_effects.csv,
              m_window_config.json, m_window_summary.txt
tests/        test_sieve.py, test_ideal_chain.py, run_all_tests.py
```

## Reproduce

```bash
pip install -r requirements.txt
python tests/run_all_tests.py            # 19 tests, ~10 s
python scripts/run_m_window_re.py        # Table I from the released trajectories
python scripts/run_composition_test.py   # Table IV
python scripts/run_sq_channel_test.py    # Tables II-III and S(q)
python scripts/plot_figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every random draw is seeded. The drift-free-window trajectories use `zlib.crc32` seeds, so they
are reproducible bit for bit.

## Related work

- PDTP v4.0, the model and the exact readout theory: DOI [10.5281/zenodo.22944122](https://doi.org/10.5281/zenodo.22944122)
- *Collapse, Convergence, and the Scale of the Prime-Gap Imprint*: DOI [10.5281/zenodo.23076433](https://doi.org/10.5281/zenodo.23076433)
- *Localized Bost–Connes Systems*: DOI [10.5281/zenodo.22883179](https://doi.org/10.5281/zenodo.22883179)

## License

MIT.
