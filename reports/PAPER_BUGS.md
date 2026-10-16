# Known issues: code bugs, paper misprints and open questions

This file collects everything found while checking bahamas against the two papers it implements:

* **Klein et al.**: R. Buscicchio, A. Klein, V. Korol, F. Di Renzo, C. J. Moore, D. Gerosa, A. Carzaniga,
  *Test for LISA foreground Gaussianity and stationarity: galactic white-dwarf binaries*, Eur. Phys. J. C 85, 887
  (2025), [arXiv:2410.08263](https://arxiv.org/abs/2410.08263). It derives the analytic Galactic envelope (App. A).
* **Pozzoli et al.**: R. Buscicchio, F. Pozzoli, D. Chirico, A. Sesana, *The first year of LISA Galactic foreground*,
  [arXiv:2511.03604](https://arxiv.org/abs/2511.03604). It defines the chunked (STFT) pipeline and its Whittle and
  Gamma likelihoods.

Both papers share their first author, so they are named here by the second author. Equation numbers are those of
the arXiv PDFs. TeX line numbers refer to `main.tex` of the arXiv sources. Checks were made on Oct 12–13, 2026.

## Summary

| ID | issue | where | status |
|---|---|---|---|
| F1 | `cosh` that should be `sinh` in H_S;(3) | code, envelope | **fixed** |
| F2 | missing $e^{-2\sigma_+^2}$, $e^{-6\sigma_+^2}$ damping on 48 terms | code, envelope | **fixed** |
| B1 | ERYN Gamma likelihood uses the wrong data term; its maximum is not at $S = \hat P$ | code, `setting_eryn.py` | **fixed** |
| B2 | Gamma shape $n_b/2$ should be $n_b$; Whittle carries the same factor ½; posteriors ≈√2 too wide | code and Pozzoli et al. Eqs. (3), (9) | **fixed** in the code; `legacy_dof` reproduces the paper; Q1 still open |
| B3 | instrument noise differs from Pozzoli et al. Eqs. (16)–(17), by up to 7% in band | code, `psd_noise.py` | **option added** (`paper_eqs`); default unchanged until Q5 is answered |
| B3b | A/E test-mass transfer function differs from Nam et al. (2023) Table II; ≤0.3% on the total noise PSD in band | code, `psd_noise.py` | **fixed** |
| B4 | gapped data with linear binning: `N` uses the whole `start` array | code, `bahamas_data.py` | **fixed** |
| B5 | gapped segments can run up to 10% into the next gap | code, `gaps.py` | **fixed** |
| B6 | the α₀/β₀ rotation is applied to envelope amplitudes/powers | code, envelope | **fixed** (was latent: inactive at α₀ = β₀ = 0) |
| B7 | point-source chunk average: the n = 4 term of A² − E² uses 16 l̄ instead of 4 l̄; A and E off by up to 8% | code, `average_envelope_num.py` | **fixed** |
| M1 | −240 should be −2430 in H_D;(4) | Klein et al. | misprint; the code is right |
| M2 | `4φ_L` should be `4Δφ_L` in H_D;(4) | Klein et al. | misprint; the code is right |
| M3 | "sin Δ3φ_L" should be sin 3Δφ_L in H_S;(3) | Klein et al. | misprint; the code is right |
| M4 | $\mathcal F_k = \lbrace m \mid \dots\rbrace$ should be $\mathcal F_m = \lbrace k \mid \dots\rbrace$ | Pozzoli et al. Eq. (7) | misprint |
| Q1–Q8 | open questions for Riccardo | | see [below](#questions-for-riccardo) |
| C1–C4 | conventions that are not errors | Klein et al. | see [below](#conventions-not-errors) |

B1–B7 were fixed on Oct 16, 2026, after commit `d0d5642`; each section below keeps the original description (line
numbers as before the fix) and ends with a **Fixed** paragraph giving the change and how it was verified. Regression
tests: [tests/test_paper_bugs.py](../tests/test_paper_bugs.py) (11 of its 14 tests fail on the pre-fix code).

The envelope errors F1 and F2 were in the code up to commit `4db1cc9` and were fixed in `66a908d`. The first wavelet runs 1
and 2 (Oct 11) used the pre-fix code. Both were redone with the fixed code on Oct 16
([WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md), [WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md)).

---

## Fixed bugs (Galactic envelope)

The analytic envelope, `envelopes_gaussian` ([modulation.py](../bahamas/psd_response/modulation.py)), and its chunk
average, `average_envelopes_gaussian` ([average_envelope.py](../bahamas/psd_response/average_envelope.py)),
implement App. A of Klein et al. Every term of the code was checked against the paper.

### How it was checked

[data/needlet_patches/paper_terms_check.py](../data/needlet_patches/paper_terms_check.py), against the arXiv TeX
source of the paper. The point-source coefficients and the Gaussian averaging rule are taken from the paper itself,
so each printed term can be recomputed independently:

1. The paper's point-source coefficients $h_{S;(n)}(b)$, $h_{D;(n)}(b)$ (App. A.1) against the paper's own detector
   model (arm vectors $\hat L_i(t)$, A = (Z − X)/√2, E = (X − 2Y + Z)/√6, inclination average 56/15 C² + 8/3 S²,
   numerical polarisation average): they agree to 10⁻¹⁴.
2. Each printed group $H_{S;(n)}$, $H_{D;(n)}$ (App. A.2) against the Gaussian average of $10240\thinspace h(b)\cos b$:
   a Fourier series in b, then the paper's characteristic function of the (b, l) Gaussian.
3. The code, statement by statement (each `Sum += …` / `Diff += …` is one group), against both.

All comparisons are made per group and per b-harmonic (the cos b, cos 3b and cos 5b terms), over 200 random
parameter sets. That is 45 terms: 5 H_S groups and 9 H_D groups, each with three harmonics.

| | terms that agree with the derivation (to 10⁻¹⁴) |
|---|---|
| paper as printed | 44 of 45 (H_D;(4) cos 3b is misprint M1) |
| code before the fix (`4db1cc9`) | 17 of 45 |
| code after the fix | 45 of 45 |

A second check, [envelope_check.py](../data/needlet_patches/envelope_check.py), compares the envelope with a
first-principles low-frequency response from the bahamas orbits. It tests a point source (with the paper's cos b sky
measure, see C1) and an exact numerical Gaussian sky average:

| | point source, A and E | parity violation | Gaussian sky average (pack-2 widths, 1 yr) |
|---|---|---|---|
| before the fix | 0.9% | 1% | 5.3% |
| after the fix | 0.5% | 2×10⁻¹⁵ | 0.4% |

The remaining 0.4–0.5% comes from the orbit models: the paper uses a rigid constellation, bahamas an eccentric orbit.

Line numbers below refer to `modulation.py` as it is now; in `average_envelope.py` the same lines are 12 further down.

### F1. `cosh` instead of `sinh` in H_S;(3)

[modulation.py:159](../bahamas/psd_response/modulation.py#L159), the −864 term. The paper prints

```math
-864\left(\cosh\frac{3\sigma_s^2}{2}\sin b_M\cos 3\Delta\phi_L + \sinh\frac{3\sigma_s^2}{2}\cos b_M\sin 3\Delta\phi_L\right)
```

```diff
-    Sum += fact3DphiL * (-864. * (cosh3HalvesSigmaSqSin * c3DphiL * sbM + cosh3HalvesSigmaSqSin * s3DphiL * cbM)
+    Sum += fact3DphiL * (-864. * (cosh3HalvesSigmaSqSin * c3DphiL * sbM + sinh3HalvesSigmaSqSin * s3DphiL * cbM)
```

At zero width (a point source) cosh = 1 and sinh = 0. The typo therefore left a term $\sin 3\Delta\phi_L \cos b_M$
that changes sign under $\hat n \to -\hat n$. The A/E auto-power must be the same in opposite directions, and the
typo broke this by 1%.

### F2. Missing damping on the 3b and 5b harmonics

In every group the paper multiplies the cos 3b / sin 3b terms by $e^{-2\sigma_+^2}$ and the cos 5b / sin 5b terms by
$e^{-6\sigma_+^2}$, e.g. H_S;(1):

```math
-32112\, e^{-2\sigma_+^2}\left(\cosh\frac{3\sigma_s^2}{2}\sin 3b_M\cos\Delta\phi_L + \sinh\frac{3\sigma_s^2}{2}\cos 3b_M\sin\Delta\phi_L\right)
```

The code had these factors (`fact3b`, `fact5b`) only in H_S;(0) (line 146) and H_D;(4) (line 172). It dropped
them in the other 48 terms: 8 lines of `Sum` (lines 149–167) and 16 of `Diff` (lines 174–205), each line with a
cosh and a sinh term. The fix puts them on the
coefficient, as in the paper:

```diff
-                         - 32112. * (cosh3HalvesSigmaSqSin * cDphiL * s3bM + sinh3HalvesSigmaSqSin * sDphiL * c3bM)
-                         - 720. * (cosh5HalvesSigmaSqSin * cDphiL * s5bM + sinh5HalvesSigmaSqSin * sDphiL * c5bM))
+                         - 32112. * fact3b * (cosh3HalvesSigmaSqSin * cDphiL * s3bM + sinh3HalvesSigmaSqSin * sDphiL * c3bM)
+                         - 720. * fact5b * (cosh5HalvesSigmaSqSin * cDphiL * s5bM + sinh5HalvesSigmaSqSin * sDphiL * c5bM))
```

and likewise for every term with `s3bM`, `c3bM`, `s5bM` or `c5bM`.

The factors are required by the Gaussian average. A term with k harmonics in b and m in l is damped by the
characteristic function, and going from k = 1 to k = 3 (or 5) multiplies it by $e^{-2\sigma_+^2}$ (or
$e^{-6\sigma_+^2}$), whatever m is. The code applied the k = 1 damping to all three harmonics. The error grows
with the widths and vanishes at zero width, which is why the point-source checks did not catch it.

### Tests

[tests/test_average_envelope.py](../tests/test_average_envelope.py) pinned the pre-fix output (2.08448). It now pins
2.56285, which the independent derivation reproduces to 10⁻¹⁰. The existing tests in
[tests/test_wavelet_model.py](../tests/test_wavelet_model.py), which check that the chunk average equals the time
average of the instantaneous envelope, pass with both files fixed. As a further check, the analytic chunk average
matches a brute-force mean of $A^2(t)$ over 20,000 samples to four digits for all 12 monthly chunks of a year.

### Impact of F1 and F2

At the pack-2 values (sin b_M −0.0966, l_M −1.626, s1 0.0416, s2 0.1384, sin ψ −0.83) over 30 days
([envelope_check.py](../data/needlet_patches/envelope_check.py)):

* The pre-fix envelope differs from the fixed one by 15% in the mean of $P_A$ (3% in $P_E$), and in shape by up to
  19% (A) and 9% (E).
* The pre-fix model can mimic the correct envelope to 0.7%, but only with shifted sky parameters. Relative to the
  run-2 OG posterior widths:

  | parameter | true | pre-fix best match | shift |
  |---|---|---|---|
  | s1 | 0.042 | 0.100 (prior edge) | 2.2σ |
  | s2 | 0.138 | 0.260 | 2.2σ |
  | sin ψ | −0.83 | −0.95 | 0.9σ |
  | lat, long | | | ≤ 0.4σ |

  Fitting real or independently simulated data with the pre-fix code would therefore overestimate the Galaxy's
  widths.
* **The first runs 1 and 2 (Oct 11)** used the pre-fix code for simulation and for both inferences (wavelet and OG).
  They were internally consistent, but run 2's injected "true sky envelope" was not the envelope of a Gaussian
  Galaxy with the stated parameters. Both were redone with the fixed code on Oct 16.
* **Pozzoli et al.** uses bahamas and generates its data from the same model it fits, so its injection/recovery
  results are self-consistent in the same way. Its comparison with the independent Yorsh 1b simulation, and any
  physical reading of σ₁², σ₂² and ψ, would be affected if its code had these errors (Q7).

---

## Code bugs B1–B7 (fixed Oct 16, 2026)

Line numbers in the descriptions refer to the code before the fix (commit `d0d5642`); the **Fixed**
paragraphs give the current ones.

### B1. ERYN Gamma likelihood: wrong data term (critical)

[setting_eryn.py:138-143](../bahamas/method/setting_eryn.py#L138-L143):

```python
log_likelihood += jnp.sum(
    (dof[j] / 2 - 1) * jnp.log(jnp.abs(tdi) ** 2)
    - (dof[j] / 2) * jnp.log(psd_model)
    - jsc.special.gammaln(dof[j] / 2)
    - (jnp.abs(tdi) ** 2) / (2 * psd_model)
)
```

For the Gamma likelihood, `tdi` is already the averaged periodogram $\hat P$, which is a power. The function squares
it again, and the last term also lacks the factor $n_b$. Per bin it evaluates

```math
-\frac{n_b}{2}\ln S - \frac{\hat P^{2}}{2S}
\qquad\text{instead of}\qquad
-\frac{n_b}{2}\ln S - \frac{n_b}{2}\frac{\hat P}{S}.
```

$\hat P^2/S$ has units of a PSD and is about 10⁻⁴⁰ for LISA, so the data term is effectively zero. The
log-likelihood reduces to $-\tfrac{n_b}{2}\sum\ln S$, which increases without bound as $S \to 0$. On a test grid
(n_b = 30, $\hat P$ = 10⁻⁴⁰) it peaks at $S = 10^{-4}\hat P$, the edge of the grid, against $S = \hat P$ for
Pozzoli et al. Eq. (9). **Any ERYN run with `likelihood: Gamma` is invalid:** the amplitudes are pushed to the lower
edge of their priors. The HMC and nessai versions ([setting_hmc.py:112](../bahamas/method/setting_hmc.py#L112),
[setting_nessai.py:77](../bahamas/method/setting_nessai.py#L77)) are correct up to B2.

**Fixed:** [setting_eryn.py:136-142](../bahamas/method/setting_eryn.py#L136-L142) now uses the same expression as
`setting_hmc.gamma_lik`: `(dof/2 - 1)*log(tdi)` and `- (dof/2) * tdi / psd_model`. Verified on a grid of $S$ with
30 bins at $\hat P$ = 10⁻⁴⁰: the maximum is at $S = \hat P$ and the curvature in $\ln S$ is −900 = −n_b × 30 bins, the
same as HMC and nessai.

### B2. Degrees of freedom of the Gamma and Whittle likelihoods

All three samplers implement Pozzoli et al. Eq. (9) as printed:
$\hat P \sim \mathrm{Gamma}(\text{shape } n_b/2,\thickspace \text{scale } 2S/n_b)$ ([setting_hmc.py:112](../bahamas/method/setting_hmc.py#L112),
[setting_nessai.py:77](../bahamas/method/setting_nessai.py#L77), and B1 once fixed). This assumes that each fine bin
carries **one** real Gaussian degree of freedom. A Fourier amplitude is complex, and its real and imaginary parts are
independent with variance $S/2$ each. So $\lvert\tilde d_k\rvert^2 = \tfrac{S}{2}\chi^2_2$, which is Gamma with shape 1
(as Klein et al. Eq. (21) states). The mean of $n_b$ of them is

```math
\hat P = \frac{S}{2n_b}\,\chi^2_{2n_b} \sim \mathrm{Gamma}\left(\text{shape } n_b,\ \text{scale } \frac{S}{n_b}\right),
\qquad \mathrm{Var}\,\hat P = \frac{S^2}{n_b},
```

whereas Eq. (9) implies $\mathrm{Var}\thinspace \hat P = 2S^2/n_b$. The first wavelet run 2 (Oct 11)
measured $n_b\thinspace \mathrm{Var}(\hat P/S) = 1.04 \pm 0.04$ on its simulated data, confirming shape $n_b$; the
bahamas simulator gives the same (see **Fixed** below). The θ-dependent part of Eq. (9) is exactly one half of the
correct one:

```math
\ln\mathcal L_{\rm Eq.(9)}(\theta) = \tfrac12 \ln\mathcal L_{\rm exact}(\theta) + \text{const}.
```

This is tempering at inverse temperature ½. The posterior peak is unchanged, but **every posterior width is about √2
too large**, and log Bayes factors are roughly halved. Equivalently, the posteriors correspond to half the data.

The Whittle likelihood, Pozzoli et al. Eq. (3), and the code
([setting_hmc.py:86](../bahamas/method/setting_hmc.py#L86), [setting_nessai.py:50](../bahamas/method/setting_nessai.py#L50),
[setting_eryn.py:78-82](../bahamas/method/setting_eryn.py#L78-L82)) use $-\tfrac12(\lvert\tilde d\rvert^2/S + \ln S)$
per complex amplitude. The exact complex Whittle likelihood is $-(\lvert\tilde d\rvert^2/S + \ln S)$. So the Whittle
and Gamma likelihoods agree with each other, as Pozzoli et al. find, but are both tempered by ½.

**Fixed:** `dof` now counts real Gaussian degrees of freedom:
[setting_inference.py:200-204](../bahamas/method/setting_inference.py#L200-L204) passes `dof = 2*count` (two per complex
amplitude), so the unchanged Gamma formulas (shape `dof/2`) have shape $n_b$. The three `whittle_lik`
([setting_hmc.py:88-89](../bahamas/method/setting_hmc.py#L88-L89), [setting_nessai.py:51-52](../bahamas/method/setting_nessai.py#L51-L52),
[setting_eryn.py:79-80](../bahamas/method/setting_eryn.py#L79-L80)) use the same `dof` (ν per amplitude):

```math
\ln\mathcal L_W = -\frac12\sum_k \nu\left(\frac{\lvert\tilde d_k\rvert^2}{S_k} + \ln S_k + \ln\frac{2\pi}{\nu}\right),
```

which for ν = 2 is the circular complex Gaussian density $-\sum_k(\lvert\tilde d_k\rvert^2/S_k + \ln \pi S_k)$ and for
ν = 1 is Eq. (3) as printed. Setting `inference: legacy_dof: true` passes `dof = count` and reproduces the paper's
Eqs. (3) and (9) exactly, including the constants, for comparisons with published results. Runs that used
`beta: 2` as a workaround should go back to `beta: 1`.

Verified: on a grid of $S$, every Gamma and Whittle version peaks at $S = \hat P$, with curvature $-n_b$ per bin
(−n_b/2 with `legacy_dof`); the fixed Whittle value equals the complex Gaussian density, and the legacy value
equals the old code's, to 10⁻¹². Data from the bahamas simulator itself (gapped, `lin` and `log`) give
$n_b\thinspace\mathrm{Var}(\hat P/S)$ = 0.94–1.10, i.e. shape $n_b$. The wavelet likelihood is unchanged.
Posteriors are now ≈√2 narrower than with the paper's likelihood, and evidences change.

### B3. Instrument noise differs from Pozzoli et al. Eqs. (16)–(17)

[psd_noise.py:175-193](../bahamas/psd_strain/psd_noise.py#L175-L193) (`psd_AE`; `psd_T`, `acc_noise` and `oms_noise`
are the same):

* **TM noise** lacks the high-frequency factor $[1 + (f/8\thinspace\mathrm{mHz})^4]$ of Eq. (16).
* **OMS noise** uses a filtered, discrete-time form instead of Eq. (17). It equals Eq. (17) for $f \ll f_s$ and drops
  below it at high frequency (−4% at 29 mHz).

Effect on the A/E noise PSD at the injected A = 2.4, P = 7.9, relative to the paper's equations with the same
transfer functions:

| f (mHz) | 0.1–0.7 | 1.7 | 4.4 | 11 | 29 |
|---|---|---|---|---|---|
| code / paper | 1.000 | 0.998 | 0.966 | 0.943 | 0.927 |

Data and model both use the code's version, so injection/recovery runs are self-consistent. The difference matters
when comparing with the paper or with data simulated elsewhere (e.g. Yorsh). Which form the paper's runs used is
question Q5.

**Option added:** the default is unchanged (existing runs stay reproducible). A fixed parameter `paper_eqs` in the
`instr_noise` source switches to Eqs. (16)–(17) exactly
([psd_noise.py:44-72](../bahamas/psd_strain/psd_noise.py#L44-L72), [psd_noise.py:269-273](../bahamas/psd_strain/psd_noise.py#L269-L273)):

```yaml
instr_noise:
  - name: "A"
    bounds: [1, 5]
    injected: 2.4
  - name: "P"
    bounds: [3, 10]
    injected: 7.9
  - name: "paper_eqs"
    injected: 1
```

It applies to the `acc_noise` and `oms_noise` sources too. With it, `noise()` equals Eqs. (16)–(17) times the Nam et
al. transfer functions to 10⁻¹⁰ in A, E and T. Once Q5 is answered, the default can be switched if needed.

### B3b. A/E test-mass transfer function differs from Nam et al. (2023)

Pozzoli et al. do not print the TDI transfer functions; they cite Nam et al., PRD 108, 082004 (2023),
[arXiv:2211.02539](https://arxiv.org/abs/2211.02539), Table II (A, E, T, uncorrelated noises, page 13). Checked on
Oct 16, 2026 against the arXiv TeX source.

**Conventions.** Table II is TDI 2.0, with $C_{XX} = 16\sin^2 w\sin^2 2w$ and $w = 2\pi f L$. The code writes the
TDI 1.0 form with `Cxx = 8 sin²w` and, for `gen2=True`, multiplies by the TDI 2.0 factor
$\lvert 1 - e^{-4iw}\rvert^2 = 4\sin^2 2w$. So `Cxx` × 4 sin²2w = 2 $C_{XX}$, and the code's coefficients are half
the table's. **The `gen2` factor 4 sin²2w is correct; do not change it to 2 sin²2w**, which would only relabel
$C_{XX}$ and halve every TDI-2 noise PSD.

| row (uncorrelated) | code, × 4 sin²2w | Nam et al. Table II | |
|---|---|---|---|
| OMS (readout ISI/RFI), A&E | 2 C_XX (2 + cos w) | 2 C_XX (2 + cos w) | ✓ |
| OMS, T | 4 C_XX (1 − cos w) | 4 C_XX (1 − cos w) | ✓ |
| TM acceleration, T | 8 C_XX (1 − cos w)² | 32 C_XX sin⁴(w/2) | ✓ (identical) |
| TM acceleration, A&E | **8 C_XX (1 + 2 cos w)** | **4 C_XX (3 + 2 cos w + cos 2w)** | ✗ |

With cos 2w = 2cos²w − 1 the table's TM A&E entry is 8 C_XX (1 + cos w + cos²w), so the code has `2 cos w` where
it should have `cos w + cos²w`. The two agree only where cos w ∈ {0, 1}, i.e. they share the low-frequency limit
(24 C_XX), and differ in shape above it. No overall factor fixes this. Affected lines:
[psd_noise.py:125](../bahamas/psd_strain/psd_noise.py#L125) (`transfer_functions_acc`) and
[psd_noise.py:189](../bahamas/psd_strain/psd_noise.py#L189) (`psd_AE`, the one used by the likelihood via `noise()`).

| f (mHz) | 0.1 | 1 | 4.4 | 10 | 20 | 29 | 40 |
|---|---|---|---|---|---|---|---|
| TM A/E transfer, code / Nam | 1.000 | 1.0005 | 1.009 | 1.044 | 1.142 | 1.052 | 0.019 |
| total A/E noise PSD, code / with Nam TM | 1.000 | 1.0004 | 1.003 | 1.001 | 1.000 | 1.000 | 1.000 |

OMS noise dominates where the transfer functions differ, so the effect on the total PSD is at most 0.3% in band, much
smaller than B3. Data and model use the same function, so injection/recovery is self-consistent. Above about 40 mHz
the code's TM term becomes negative (1 + 2 cos w < 0); this is outside the current band but would matter if `f2` were
raised.

**Fixed:** in both places ([psd_noise.py:141](../bahamas/psd_strain/psd_noise.py#L141) and
[psd_noise.py:205](../bahamas/psd_strain/psd_noise.py#L205)),

```python
tr_acc_AE = Cxx * 4*(1 + jnp.cos(w) + jnp.cos(w)**2)   # = Cxx * 2*(3 + 2cos w + cos 2w), Nam et al. Table II
```

The T-channel and OMS expressions needed no change. All four TDI-2 transfer functions now equal Table II to 10⁻¹⁰ over
0.1–35 mHz (test `test_noise_transfer_functions_nam`). The default noise PSD changes by at most 0.3% in band.

### B4. Gapped data with linear binning: `N` is an array

[bahamas_data.py:191](../bahamas/bahamas_data.py#L191):

```python
N = (self.end[i] - self.start) // self.dt      # should be self.start[i]
```

`N_tot[i]` becomes an array of all segment lengths minus one start time. It is only used for `mod: 'lin'`
(`dof = self.N_tot[i] // self.nseg`, line 287), where it breaks the bin size. With `mod: 'log'`, the default in the
templates, it is unused. A related oddity: in `lin` mode `dof = N // nseg` counts *time* samples, so a chunk of
$N/2$ frequencies yields about `nseg/2` bins, not `nseg`.

Running the gapped `lin` path end to end shows that the first part is not silent: `average_chunks` receives an array of
section counts, some negative, and crashes (`TypeError: slice indices must be integers`).

**Fixed:**

* [bahamas_data.py:191](../bahamas/bahamas_data.py#L191) uses `self.start[i]`.
* In `lin` mode, [bahamas_data.py:286-294](../bahamas/bahamas_data.py#L286-L294) splits the positive frequencies
  (the DC amplitude is zero) into `nseg` bins of `len(freqs > 0) // nseg` amplitudes. `average_chunks` now also
  returns the number of amplitudes actually averaged in each bin
  ([setting_data.py:60](../bahamas/method/setting_data.py#L60)), which is stored as `count`. Chunked `lin` runs
  therefore get `nseg` bins instead of about `nseg/2`.
* The PSD-to-amplitude factor uses each segment's own length
  ([bahamas_data.py:273](../bahamas/bahamas_data.py#L273)), needed once gapped segments can be shorter than
  `sched_period` (B5).

Verified with `SignalProcessor` on gapped data: `lin` runs, gives 50 bins for `nseg: 50`, mean $\hat P/S$ = 0.99 and
$n_b\thinspace\mathrm{Var}(\hat P/S)$ = 1.06–1.10 against the bin-averaged injected PSD.

### B5. Gapped segments can extend into the next gap

[gaps.py:68-69](../bahamas/method/gaps.py#L68-L69): with unscheduled gaps, a stretch of data between two gaps is kept
if it is longer than 0.9 × `sched_period`, but the segment is then set to exactly `sched_period` from its start. A
kept stretch shorter than `sched_period` therefore runs up to 10% into the following gap. In simulations this only
means slightly more data than the gap pattern allows; on real data, the segment would include gap samples. Separately,
when the gaps exceed the duty cycle, the loop removes gaps from the *end* of the list, so the gap pattern is not
uniform over the year. Two smaller issues in the same function: the loop over stretches stops one gap early, so the
stretch after the last gap is never used, and without unscheduled gaps the last segment can end after `T_obs` (day
71.5 of a 60-day run in one test).

**Fixed** ([gaps.py:53-87](../bahamas/method/gaps.py#L53-L87)):

* A kept stretch is cut to `min(start + sched_period, stretch end)`, so segments never contain gap samples.
* The stretches before the first and after the last gap are considered too.
* Without unscheduled gaps, segments are clipped at `T_obs` and kept only if longer than 0.9 × `sched_period`.
* When the gaps exceed the duty cycle, randomly chosen gaps are removed instead of the last ones. A run with a given
  seed then has a different gap pattern from before, but only when this branch is reached.
* If no stretch survives, the function raises a clear `ValueError` instead of an `IndexError`.

Segments can now be shorter than `sched_period`, so the normalisation uses each segment's own length (B4). Verified
over 20 seeds (1 yr, 14-day period, 3 h unscheduled gaps every ~10 days): before, 28 of 144 segments contained gap
samples; now none do, with the same data per year (101 vs 102 days). A 3.6-day segment of a 4-day period gives mean
$\hat P/S$ = 0.99; the old fixed normalisation would give 0.9.

### B6. α₀/β₀ rotation applied to amplitudes and powers (latent)

[modulation.py:210](../bahamas/psd_response/modulation.py#L210) and
[average_envelope.py:222](../bahamas/psd_response/average_envelope.py#L222) apply the rotation
$A = \lvert\cos 2(\beta_0-\alpha_0)A_0 - \sin 2(\beta_0-\alpha_0)E_0\rvert$ to envelope amplitudes (`modulation.py`)
and to chunk-averaged powers (`average_envelope.py`). Klein et al. App. A apply this rotation to the *signals*
$h_A$, $h_E$. The power envelope of the rotated channels then needs the A–E cross term, which is not computed. This
has no effect while α₀ = β₀ = 0, as everywhere in bahamas and in both papers, but non-zero values would give wrong
envelopes. The same rotation was also in `average_envelope_num.py` (lines 191–195 before the fix),
used by `galactic_DWD_num`.

**Fixed:** the rotation does not need the cross term after all. Rotating $(h_A, h_E)$ by $2\chi$ with
$\chi = \beta_0 - \alpha_0$ (Klein et al. TeX lines 694–697) leaves $A^2 + E^2$ unchanged and shifts the phase of
$A^2 - E^2$:

```math
\sum_n h_{D;(n)} \cos\left(n\Delta\phi_L + 4\bar l\right) \;\to\; \sum_n h_{D;(n)} \cos\left(n\Delta\phi_L + 4\bar l - 4\chi\right).
```

This was checked against the paper's detector model with the rotation applied to the signals (point source,
inclination and polarisation averaged, 100 random $b$, $l$, $t$, $\chi$): agreement to 10⁻¹⁴. Since $\chi$ is a
constant, the shift passes through the Gaussian sky average unchanged. The code now subtracts $4(\beta_0 - \alpha_0)$
from `FourphiMbar` ([modulation.py:86](../bahamas/psd_response/modulation.py#L86),
[average_envelope.py:98](../bahamas/psd_response/average_envelope.py#L98),
[average_envelope_num.py:166](../bahamas/psd_response/average_envelope_num.py#L166)) and no longer rotates the
envelopes. The $l \to l - \alpha_0$ shift was already there.

Verified:

* Point source, random α₀ and β₀, against the detector model with the paper's rotation: 2×10⁻¹⁴ (pre-fix code: errors
  up to a factor 6). This includes the fixed π/6 offset of C3.
* The chunk average equals the time average of the instantaneous envelope at the pack-2 widths with α₀, β₀ ≠ 0 to
  7×10⁻⁹ (pre-fix: 110%).
* At α₀ = β₀ = 0 the output is unchanged to 10⁻¹⁴.

---

### B7. Point-source chunk average: wrong phase in one A² − E² term

`envelopes_time_average` in [average_envelope_num.py](../bahamas/psd_response/average_envelope_num.py) is the
chunk-averaged envelope of a point source, summed over a sky grid with galaxy-model weights by `galactic_DWD_num`
([psd_galaxy.py:391](../bahamas/psd_strain/psd_galaxy.py#L391)). Found while checking B6; it is not in either paper's
pipeline (`galactic_DWD_time` uses the Gaussian envelope).

The n = 4 term of A² − E² is $h_{D;(4)}\cos(4\Delta\phi_L + 4\bar l)$ with $\Delta\phi_L = \Omega t - a$, whose average
over $[t_1, t_2]$ is

```math
\frac{\sin(4a - 4\bar l - 4\Omega t_1) - \sin(4a - 4\bar l - 4\Omega t_2)}{4\Omega T}.
```

`FourphiNbar` already holds $4\bar l$, but line 179 used `jnp.sin(4. * (a - FourphiNbar - phit1))`, i.e.
$\sin(4a - 16\bar l - 4\Omega t_1)$. The other terms are written correctly (e.g. `sin(3*a - FourphiNbar - 3*phit1)`).
A² + E² is unaffected; A and E individually were off by up to 8% (30-day chunks, random sky positions).

**Fixed:** [average_envelope_num.py:179](../bahamas/psd_response/average_envelope_num.py#L179) now uses
`jnp.sin(4. * a - FourphiNbar - 4. * phit1) - jnp.sin(4. * a - FourphiNbar - 4. * phit2)`. Verified against the time
average of the point-source envelope from `envelopes_gaussian` at zero width (divided by cos b, C1), which is itself
checked term by term against the paper: ratios 1.000 in all 12 cases, against 0.921–1.061 before (test
`test_point_source_time_average`). Only runs with the `galactic_DWD_num` source are affected.

## Misprints in the papers (the code is right)

M1–M3 are in Klein et al. App. A.2; M4 is in Pozzoli et al. Sec. II.B.

**M1. H_D;(4), the cos 3b coefficient** (TeX line 880). Printed:

```math
e^{-\frac{1}{4}\sigma_S^2-\frac{1}{4}\sigma_c^2}\left(324\cos b_M - 240\, e^{-2\sigma_+^2}\cos 3b_M - 5670\, e^{-6\sigma_+^2}\cos 5b_M\right)\cos\left(4\phi_L+4\bar l_M\right)
```

The Gaussian average of the paper's own $h_{D;(4)}$ gives **−2430**, not −240. The code has −2430.

**M2. H_D;(4), the phase** (same line). Printed $\cos(4\phi_L + 4\bar l_M)$; $\phi_L$ on its own is not defined in the
paper. The derivation gives $\cos(4\Delta\phi_L + 4\bar l_M)$, with $\Delta\phi_L = \Omega t - l_M$, as for every other
H_D;(n). The code uses `c4DphiL_4phiMbar`, i.e. $\cos(4\Delta\phi_L + 4\bar l_M)$.

**M3. H_S;(3), the 3b and 5b terms** (TeX lines 869–870). Both print "sin Δ3φ_L" where they mean sin 3Δφ_L.

**M4. Pozzoli et al. Eq. (7)**. Printed $\mathcal F_k = \lbrace m \mid f_m \lt f_k \lt f_{m+1}\rbrace$, while the
text and Eq. (8) use $\mathcal F_m$ as the set of fine frequencies in coarse bin $m$. The intended definition is
$\mathcal F_m = \lbrace k \mid f_m \lt f_k \lt f_{m+1}\rbrace$.

---

## Questions for Riccardo

**Q1. Degrees of freedom (B2).** Is the shape $n_{b_m}/2$ in Eq. (9), and the ½ in Eq. (3), intentional, e.g. a
convention in which each complex amplitude counts as one degree of freedom? With $\langle\lvert\tilde d\rvert^2\rangle = S$
(Eq. 2), the exact shape is $n_{b_m}$, and the published posteriors would be about √2 too wide. The code now uses
the exact likelihood by default; `inference: legacy_dof: true` reproduces Eqs. (3) and (9) as printed.

**Q2. Frequency band.** Pozzoli et al. quote $f_{\max} = 2.9$ mHz. The bahamas template uses `f2: 0.029` (29 mHz),
and the GF knee lies near a few mHz, so 2.9 mHz looks like a misprint for 29 mHz. Which value was used?

**Q3. Observation-time drift (Eqs. 12–13).** In the sequential analysis, is each new chunk simulated with
$T_{\rm obs}$ equal to the cumulative time up to that chunk? If so, $f_1$ and $f_{\rm knee}$ differ from chunk to
chunk in the data, while the inference samples a single $\log_{10} f_1$ and $\log_{10} f_{\rm knee}$. Does the model
apply Eqs. 12–13 per chunk, or is the recovered value an effective average? bahamas currently uses one `Tobs` (or a
fixed `fknee`, `fr1`) for all chunks, so the drift is not implemented.

**Q4. TDI generation.** Pozzoli et al. state first-generation TDI. The bahamas templates have `gen2: True`, which
multiplies both noise and GF by $4\sin^2(4\pi f L)$. Which was used for the paper's runs?

**Q5. Noise model (B3).** Were the paper's runs made with Eqs. (16)–(17) as printed, or with the bahamas form (no
8 mHz factor, filtered OMS)? Also, the A/E test-mass transfer function in the code does not match Nam et al. (2023)
Table II, which the paper cites (B3b, now fixed). Did the paper's runs use the code's form? The default noise
levels are unchanged; `paper_eqs` switches to Eqs. (16)–(17).

**Q6. Coarse-grid definition.** Eqs. (5)–(8) place $n_g = 1000$ log bins between $f_{\min}$ and $f_{\max}$. The code
([setting_data.py:82-83](../bahamas/method/setting_data.py#L82-L83)) places `nseg` bins between $1/T$ and the
Nyquist frequency, then keeps those inside `[f1, f2]`. For a one-month chunk only 481 of 1000 bins fall in
[0.1, 29] mHz, and they are about 1.2% wide instead of the paper's narrower bins. Which definition produced the
paper's results?

**Q7. Envelope code used for the paper.** Did the code behind Pozzoli et al. contain F1 and F2? Injection/recovery is
self-consistent either way, but the Yorsh comparison and any physical reading of σ₁², σ₂², ψ would change.

**Q8. Conventions C2 and C3.** Are the sign of the cross terms and the π/12 arm-labelling phase intended?

---

## Conventions, not errors

These do not make any result wrong, but they matter when the parameters are read physically.

* **C1. Point-source limit includes cos b.** Klein et al. normalise the Gaussian in the (b, l) plane and integrate
  with the sky measure cos b db dl. At zero width the formula therefore returns the point-source power times
  cos b_M, not the power itself.
* **C2. Sign of the cross terms.** The paper's distribution p(b, l) has covariance
  $\Sigma_{bl} = -\tfrac{1}{2}\sigma_D^2\sin 2\delta$, consistent with its printed characteristic function
  (the $+\tfrac{mn}{2}\sigma_D^2\sin 2\delta$ term). The printed groups, and the code, follow the opposite sign:
  they describe the stated Gaussian rotated by −δ instead of δ.
* **C3. Arm labelling.** With the arm vectors $\hat L_1, \hat L_2, \hat L_3$ exactly as printed (α₀ = β₀ = 0), the
  detector gives the h_D phase $4(l + \pi/4)$, not $4\bar l = 4(l + \pi/12)$. This is a 30° offset in β₀. The groups
  and the code use π/12, and the code's A and E match the bahamas-orbit response individually.
* **C4. Orientation range in the code.** The code takes `sinPsi` = sin 2δ and sets `cosPsi = +sqrt(1 - sinPsi**2)`, so
  only cos 2δ ≥ 0 can be represented. With s1 < s2 fixed by the priors, half of the ellipse orientations are out of
  reach. This should be fixed (e.g. sample 2δ directly) if the orientation is to be measured.

---

## Approximations still to investigate

* **Model at the bin midpoint.** The chunked model evaluates $S(\theta)$ at one frequency per coarse bin
  (`0.5*(first + last)`, [setting_data.py:91](../bahamas/method/setting_data.py#L91)), not averaged over the bin.
  This is negligible for smooth spectra, but not near sharp features such as the TDI-2 zero of $\sin^2(4\pi f L)$ at
  1/(4L) ≈ 30 mHz, just above the band edge.
* **Within-chunk modulation.** The stationary-chunk model discards the variation of $M^2(t)$ within a chunk. For
  monthly chunks this reaches a factor ≈2 in some months (e.g. $A^2$ from 3.7 to 6.9 in month 2 of the pack-2
  envelope). Its bias on the sky parameters as a function of chunk length has not been quantified.
* **Data kept with unscheduled gaps.** Only stretches longer than 0.9 × `sched_period` are kept. With 3 h unscheduled
  gaps every ~10 days and a 14-day period, about 100 days per year survive, far below the 75% duty cycle. This was
  so before the B5 fix as well.

## Reproduce

```
cd data/needlet_patches
../../.venv/bin/python paper_terms_check.py   # term-by-term table (paper, code before the fix, code now), ~1 min
../../.venv/bin/python envelope_check.py      # first-principles comparison and impact on the pack-2 envelope
```

`paper_terms_check.py` reads the pre-fix code from git (commit `4db1cc9`). The B1–B6 fixes are covered by

```
.venv/bin/python -m pytest tests/test_paper_bugs.py      # ~15 s
```

The likelihood checks can also be repeated by hand: evaluate each likelihood on a grid of $S$ for fixed $\hat P$.
