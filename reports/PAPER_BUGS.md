# Known issues: code bugs, paper misprints and open questions

Checked against two papers (both first-authored by R. Buscicchio, so named by the second author):

* **Klein et al.**, *Test for LISA foreground Gaussianity and stationarity*, [arXiv:2410.08263](https://arxiv.org/abs/2410.08263). App. A: analytic Galactic envelope.
* **Pozzoli et al.**, *The first year of LISA Galactic foreground*, [arXiv:2511.03604](https://arxiv.org/abs/2511.03604). Chunked pipeline, Whittle and Gamma likelihoods.

Equation numbers are those of the arXiv PDFs. The previous, longer version of this file (full derivations, impact
tables) is in git history (commit `1a1afde`).

## Summary

| ID | issue | status |
|---|---|---|
| [F1](#f1-cosh-instead-of-sinh-in-h_s3) | `cosh` should be `sinh` in H_S;(3) | fixed |
| [F2](#f2-missing-damping-on-the-3b-and-5b-harmonics) | missing $e^{-2\sigma_+^2}$, $e^{-6\sigma_+^2}$ damping on 48 terms | fixed |
| [B1](#b1-eryn-gamma-likelihood-wrong-data-term) | ERYN Gamma likelihood squares $\hat P$ | fixed |
| [B2](#b2-degrees-of-freedom-of-gamma-and-whittle) | Gamma shape $n_b/2$ should be $n_b$; posteriors ≈√2 too wide | fixed (paper form via `legacy_dof`) |
| [B3](#b3-instrument-noise-differs-from-pozzoli-et-al-eqs-1617) | noise PSD differs from Pozzoli et al. Eqs. (16)–(17) | option `paper_eqs` added |
| [B3b](#b3b-ae-test-mass-transfer-function) | A/E test-mass transfer function differs from Nam et al. | fixed |
| [B4](#b4-gapped-data-with-linear-binning) | `self.start` instead of `self.start[i]` | fixed |
| [B5](#b5-gapped-segments-run-into-the-next-gap) | gapped segments run into the next gap | fixed |
| [B6](#b6-α₀β₀-rotation-applied-to-powers) | α₀/β₀ rotation applied to powers, not signals | fixed (was inactive at α₀ = β₀ = 0) |
| [B7](#b7-wrong-phase-in-one-a²e²-term-of-the-point-source-average) | $16\bar l$ instead of $4\bar l$ in one term | fixed |
| [M1–M4](#misprints-in-the-papers-code-is-right) | misprints in the papers | code is right |

Every fix can be turned off (restores the pre-fix code exactly) via [bahamas/fixes.py](../bahamas/fixes.py):

```yaml
fixes:
  F1: false   # any of F1, F2, B1, B2, B3b, B4, B5, B6, B7
```

Use the same `fixes` section in the data and inference configs. Tests: [tests/test_paper_bugs.py](../tests/test_paper_bugs.py).

---

## Galactic envelope ([modulation.py](../bahamas/psd_response/modulation.py), same lines +12 in [average_envelope.py](../bahamas/psd_response/average_envelope.py))

### F1. `cosh` instead of `sinh` in H_S;(3)

The cross term $\cos b_M\sin3\Delta\phi_L$ of H_S;(3) was multiplied by cosh instead of sinh. At zero width this leaves
a spurious term that should vanish.

**Paper** ([Klein et al. App. A.2](https://arxiv.org/abs/2410.08263), H_S;(3)):

```math
-864\left(\cosh\tfrac{3\sigma_s^2}{2}\sin b_M\cos 3\Delta\phi_L + \sinh\tfrac{3\sigma_s^2}{2}\cos b_M\sin 3\Delta\phi_L\right)
```

**Code before** ([modulation.py:168](../bahamas/psd_response/modulation.py#L168)):

```math
-864\left(\cosh\tfrac{3\sigma_s^2}{2}\sin b_M\cos 3\Delta\phi_L + \cosh\tfrac{3\sigma_s^2}{2}\cos b_M\sin 3\Delta\phi_L\right)
```

**Fix.** The Gaussian average of $\sin kb\cos m\Delta\phi$ always pairs $\cos kb_M\sin m\Delta\phi_M$ with
$\sinh(km\,\sigma_s^2/2)$, so sinh is correct (and $\sinh 0 = 0$ recovers the point source).

```diff
-    Sum += fact3DphiL * (-864. * (cosh3HalvesSigmaSqSin * c3DphiL * sbM + cosh3HalvesSigmaSqSin * s3DphiL * cbM)
+    Sum += fact3DphiL * (-864. * (cosh3HalvesSigmaSqSin * c3DphiL * sbM + sinh3HalvesSigmaSqSin * s3DphiL * cbM)
```

Corrected equation: the paper's, as printed above.

**Test.** [paper_terms_check.py](../simulations/30_days/needlet_patches/paper_terms_check.py) matches every term against an independent Gaussian average to 10⁻¹⁴; the A/E power is now the same in opposite sky directions (it was off by 1%).

### F2. Missing damping on the 3b and 5b harmonics

In every group with $m \ge 1$ the code applied the $k = 1$ damping to the $\sin 3b$, $\cos 3b$, $\sin 5b$, $\cos 5b$ terms too,
dropping the paper's extra $e^{-2\sigma_+^2}$ and $e^{-6\sigma_+^2}$ (48 terms).

**Paper** ([Klein et al. App. A.2](https://arxiv.org/abs/2410.08263), e.g. H_S;(1)):

```math
-32112\,e^{-2\sigma_+^2}\left(\cosh\tfrac{3\sigma_s^2}{2}\sin 3b_M\cos\Delta\phi_L + \sinh\tfrac{3\sigma_s^2}{2}\cos 3b_M\sin\Delta\phi_L\right)
```

**Code before** ([modulation.py:155-215](../bahamas/psd_response/modulation.py#L155-L215)):

```math
-32112\left(\cosh\tfrac{3\sigma_s^2}{2}\sin 3b_M\cos\Delta\phi_L + \sinh\tfrac{3\sigma_s^2}{2}\cos 3b_M\sin\Delta\phi_L\right)
```

**Fix.** The Gaussian damping of a term with $k$ harmonics in $b$ is $D_{k,m}$, and
$D_{k,m}/D_{1,m} = e^{-(k^2-1)\sigma_+^2/4}$, which is $e^{-2\sigma_+^2}$ for $k = 3$ and $e^{-6\sigma_+^2}$ for $k = 5$ for every $m$.
Added to all 48 terms:

```diff
-  - 32112. * (cosh3HalvesSigmaSqSin * cDphiL * s3bM + sinh3HalvesSigmaSqSin * sDphiL * c3bM)
+  - 32112. * fact3bDphiL * (cosh3HalvesSigmaSqSin * cDphiL * s3bM + sinh3HalvesSigmaSqSin * sDphiL * c3bM)
```

Corrected equation: the paper's, as printed above.

**Test.** Same term-by-term check (45 of 45 terms now agree, 17 of 45 before), plus [test_average_envelope.py](../tests/test_average_envelope.py), which pins the new value 2.56285 (was 2.08448).

**Impact of F1 + F2** (pack-2 values, 30 days): the mean $P_A$ changes by 15%. The pre-fix model overestimates the
widths s1, s2 by about 2σ. Wavelet runs 1 and 2 were redone with the fixed code on Oct 16.

---

## Likelihoods

### B1. ERYN Gamma likelihood: wrong data term

ERYN's Gamma likelihood squared $\hat P$, which is already a power, and dropped $n_b$ in the data term. The data term then
becomes about 10⁻⁴⁰ and the likelihood runs off to $S \to 0$.

**Paper** ([Pozzoli et al. Eq. 9](https://arxiv.org/abs/2511.03604)), per bin:

```math
-\tfrac{n_b}{2}\ln S - \tfrac{n_b}{2}\tfrac{\hat P}{S}
```

**Code before** ([setting_eryn.py:147-152](../bahamas/method/setting_eryn.py#L147-L152)):

```math
-\tfrac{n_b}{2}\ln S - \tfrac{\hat P^2}{2S}
```

**Fix.** Same expression as the HMC and nessai versions.

```diff
-    (dof[j] / 2 - 1) * jnp.log(jnp.abs(tdi) ** 2)
-    - (jnp.abs(tdi) ** 2) / (2 * psd_model)
+    (dof[j] / 2 - 1) * jnp.log(tdi)
+    - (dof[j] / 2) * tdi / psd_model
```

Corrected equation: the paper's, as printed above.

**Test.** `test_gamma_peaks_at_periodogram`: the maximum is now at $S = \hat P$ (it was at the edge of the grid).

### B2. Degrees of freedom of Gamma and Whittle

A complex Fourier amplitude has 2 real degrees of freedom, so the average of $n_b$ of them is Gamma with shape $n_b$.
The paper uses shape $n_b/2$, which halves the log-likelihood and makes posteriors about √2 too wide.

**Paper** ([Pozzoli et al. Eqs. 3, 9](https://arxiv.org/abs/2511.03604)):

```math
\ln\mathcal L_W = -\tfrac12\sum_k\left(\tfrac{|\tilde d_k|^2}{S_k} + \ln S_k\right),\qquad \hat P \sim \mathrm{Gamma}\!\left(\tfrac{n_b}{2},\ \tfrac{2S}{n_b}\right)
```

**Code before** ([setting_hmc.py:86-92](../bahamas/method/setting_hmc.py#L86-L92), [setting_nessai.py](../bahamas/method/setting_nessai.py#L26), [setting_eryn.py](../bahamas/method/setting_eryn.py#L29)): the paper's form, with `dof = count`.

**Fix.** `dof` now counts real degrees of freedom: 2 per complex amplitude ([setting_inference.py:207-208](../bahamas/method/setting_inference.py#L207-L208)).

```diff
-    return log_like, data, freqs, response, config['dt'], t1, t2, count, gen2
+    dof = [2 * np.asarray(c) for c in count]
+    return log_like, data, freqs, response, config['dt'], t1, t2, dof, gen2
```

```math
\ln\mathcal L_W = -\sum_k\left(\tfrac{|\tilde d_k|^2}{S_k} + \ln \pi S_k\right),\qquad \hat P \sim \mathrm{Gamma}\!\left(n_b,\ \tfrac{S}{n_b}\right)
```

`inference: legacy_dof: true` (or `fixes: B2: false`) reproduces the paper's Eqs. 3 and 9 exactly.

**Test.** `test_whittle_is_complex_gaussian`, `test_gamma_peaks_at_periodogram`; the simulator gives $n_b\,\mathrm{Var}(\hat P/S) = 0.94$–$1.10$, i.e. shape $n_b$.

---

## Instrument noise ([psd_noise.py](../bahamas/psd_strain/psd_noise.py))

### B3. Instrument noise differs from Pozzoli et al. Eqs. (16)–(17)

The TM noise lacks the $(1 + (f/8\,\mathrm{mHz})^4)$ factor, and the OMS noise uses a filtered discrete-time form. The
A/E noise is up to 7% lower at 29 mHz.

**Paper** ([Pozzoli et al. Eqs. 16–17](https://arxiv.org/abs/2511.03604)):

```math
S_{\rm TM} = \frac{(A\,10^{-15})^2}{(2\pi f c)^2}\Big[1+\big(\tfrac{0.4\,\rm mHz}{f}\big)^2\Big]\Big[1+\big(\tfrac{f}{8\,\rm mHz}\big)^4\Big],\qquad
S_{\rm OMS} = (P\,10^{-12})^2\Big(\tfrac{2\pi f}{c}\Big)^2\Big[1+\big(\tfrac{2\,\rm mHz}{f}\big)^4\Big]
```

**Code** ([psd_noise.py:55-73](../bahamas/psd_strain/psd_noise.py#L55-L73)): $S_{\rm TM}$ without the $(f/8\,\rm mHz)^4$ factor; $S_{\rm OMS}$ filtered (equals Eq. 17 for $f \ll f_s$).

**Option, default unchanged** (pending question Q5). Add the fixed parameter `paper_eqs` to switch to the paper's equations ([psd_noise.py:45-52](../bahamas/psd_strain/psd_noise.py#L45-L52)):

```yaml
instr_noise:
  - name: "paper_eqs"
    injected: 1
```

**Test.** `test_noise_paper_eqs`: with `paper_eqs`, the noise equals Eqs. 16–17 × the Nam et al. transfer functions to 10⁻¹⁰.

### B3b. A/E test-mass transfer function

The A/E TM transfer function has `2 cos w` where Nam et al. (2023), cited by Pozzoli et al., have `cos w + cos²w`. The
effect on the total noise PSD is at most 0.3% in band.

**Paper** ([Nam et al. Table II](https://arxiv.org/abs/2211.02539), TDI 2.0, $C_{XX} = 16\sin^2 w\sin^2 2w$):

```math
4\,C_{XX}\,(3 + 2\cos w + \cos 2w) = 8\,C_{XX}\,(1 + \cos w + \cos^2 w)
```

**Code before** ([psd_noise.py:142](../bahamas/psd_strain/psd_noise.py#L142), [psd_noise.py:208](../bahamas/psd_strain/psd_noise.py#L208)):

```math
8\,C_{XX}\,(1 + 2\cos w)
```

**Fix.** Use Table II.

```diff
-    tr_acc_AE = Cxx * 4*(1 + 2*jnp.cos(w))
+    tr_acc_AE = Cxx * 4*(1 + jnp.cos(w) + jnp.cos(w)**2)
```

Corrected equation: Table II, as printed above. (The `gen2` factor $4\sin^2 2w$ is correct as it is.)

**Test.** `test_noise_transfer_functions_nam`: all four TDI-2 transfer functions equal Table II to 10⁻¹⁰.

---

## Data and gaps

### B4. Gapped data with linear binning

The segment length uses the whole `start` array instead of `start[i]`, so gapped `lin` runs crash. Not in the paper.

**Intended:** $N_i = (t^{\rm end}_i - t^{\rm start}_i)/\Delta t$.

**Code before** ([bahamas_data.py:193](../bahamas/bahamas_data.py#L193)):

```python
N = (self.end[i] - self.start) // self.dt
```

**Fix.** Index the segment. `lin` mode now also makes `nseg` bins (about `nseg/2` before) and stores the real per-bin
counts. The PSD normalisation uses each segment's own length ([bahamas_data.py:275](../bahamas/bahamas_data.py#L275)).

```diff
-    N = (self.end[i] - self.start) // self.dt
+    N = (self.end[i] - self.start[i]) // self.dt
```

**Test.** `test_average_chunks_counts`; gapped `lin` gives 50 bins for `nseg: 50` and mean $\hat P/S = 0.99$.

### B5. Gapped segments run into the next gap

A data stretch longer than 0.9 × `sched_period` was extended to a full `sched_period`, so it could run up to 10% into the
next gap. Not in the paper.

**Intended:** segment $= [t_{\rm start},\ \min(t_{\rm start} + T_{\rm sched},\ t_{\rm stretch\ end})]$.

**Code before** ([gaps.py:84-87](../bahamas/method/gaps.py#L84-L87)): segment $= [t_{\rm start},\ t_{\rm start} + T_{\rm sched}]$.

**Fix.** Clip the segment at the end of its stretch. Also use the stretches before the first and after the last gap, and
drop excess gaps at random rather than from the end.

```diff
-    data_seg.append((data_start, data_start + scheduled_period))
+    data_seg.append((data_start, min(data_start + scheduled_period, data_end)))
```

(The loop now also runs over the stretches `[0, first gap]` and `[last gap, T_obs]`.)

**Test.** `test_gap_segments_stay_between_gaps`; over 20 seeds, segments containing gap samples went from 28 of 144 to 0.

---

## Envelope rotation and point-source average

### B6. α₀/β₀ rotation applied to powers

The code rotated the envelope amplitudes and powers. The paper rotates the signals $(h_A, h_E)$. This had no effect at
α₀ = β₀ = 0, which every run uses.

**Paper** ([Klein et al. App. A](https://arxiv.org/abs/2410.08263)):

```math
(h_A, h_E) \to R\big[2(\beta_0-\alpha_0)\big]\,(h_A, h_E)
```

**Code before** ([modulation.py](../bahamas/psd_response/modulation.py), [average_envelope.py](../bahamas/psd_response/average_envelope.py)):

```math
A = \big|\cos 2(\beta_0-\alpha_0)\,A_0 - \sin 2(\beta_0-\alpha_0)\,E_0\big|
```

**Fix.** Rotating the signals leaves $A^2 + E^2$ unchanged and only shifts the phase of $A^2 - E^2$ by $-4(\beta_0-\alpha_0)$
([modulation.py:92](../bahamas/psd_response/modulation.py#L92), [average_envelope.py:104](../bahamas/psd_response/average_envelope.py#L104), [average_envelope_num.py:169](../bahamas/psd_response/average_envelope_num.py#L169)).

```diff
+    FourphiMbar = FourphiMbar - 4. * (beta0 - alpha0)
     ...
-    A = jnp.abs(cDect * A0 - sDect * E0)
-    E = jnp.abs(sDect * A0 + cDect * E0)
+    A, E = A0, E0
```

```math
\sum_n h_{D;(n)}\cos\!\big(n\Delta\phi_L + 4\bar l\big) \;\to\; \sum_n h_{D;(n)}\cos\!\big(n\Delta\phi_L + 4\bar l - 4(\beta_0-\alpha_0)\big)
```

**Test.** `test_envelope_rotation`: agrees with the paper's detector model with rotated signals to 2×10⁻¹⁴ (errors up to ×6 before).

### B7. Wrong phase in one A²−E² term of the point-source average

In the chunk average of the $n = 4$ term, `FourphiNbar` (already $4\bar l$) was multiplied by 4 again. Only the
`galactic_DWD_num` source is affected, where A and E were off by up to 8%.

**Intended**, the average of $h_{D;(4)}\cos(4\Delta\phi_L + 4\bar l)$ over $[t_1, t_2]$:

```math
\frac{\sin(4a - 4\bar l - 4\Omega t_1) - \sin(4a - 4\bar l - 4\Omega t_2)}{4\Omega T}
```

**Code before** ([average_envelope_num.py:183](../bahamas/psd_response/average_envelope_num.py#L183)): $\sin(4a - 16\bar l - 4\Omega t_1) - \dots$

**Fix.**

```diff
-    jnp.sin(4. * (a - FourphiNbar - phit1)) - jnp.sin(4. * (a - FourphiNbar - phit2))
+    jnp.sin(4. * a - FourphiNbar - 4. * phit1) - jnp.sin(4. * a - FourphiNbar - 4. * phit2)
```

**Test.** `test_point_source_time_average`: matches the time-averaged instantaneous envelope, ratio 1.000 (was 0.92–1.06).

---

## Misprints in the papers (code is right)

| ID | where | printed | correct |
|---|---|---|---|
| M1 | [Klein et al.](https://arxiv.org/abs/2410.08263) H_D;(4), cos 3b term | $-240\,e^{-2\sigma_+^2}\cos 3b_M$ | $-2430\,e^{-2\sigma_+^2}\cos 3b_M$ |
| M2 | Klein et al. H_D;(4), phase | $\cos(4\phi_L + 4\bar l_M)$ | $\cos(4\Delta\phi_L + 4\bar l_M)$ |
| M3 | Klein et al. H_S;(3), 3b and 5b terms | $\sin\Delta 3\phi_L$ | $\sin 3\Delta\phi_L$ |
| M4 | [Pozzoli et al.](https://arxiv.org/abs/2511.03604) Eq. 7 | $\mathcal F_k = \lbrace m \mid f_m < f_k < f_{m+1}\rbrace$ | $\mathcal F_m = \lbrace k \mid f_m < f_k < f_{m+1}\rbrace$ |

## Questions for Riccardo

* **Q1.** Is the Gamma shape $n_b/2$ (Eq. 9), and the ½ in Eq. 3, intentional? (B2)
* **Q2.** Is $f_{\max} = 2.9$ mHz a misprint for 29 mHz?
* **Q3.** Is $T_{\rm obs}$ in Eqs. 12–13 the cumulative time per chunk? bahamas uses one `Tobs` for all chunks.
* **Q4.** First-generation TDI (paper) or `gen2: True` (templates)?
* **Q5.** Were the runs made with Eqs. 16–17 as printed, or with the bahamas noise form? (B3, B3b)
* **Q6.** Coarse grid: 1000 log bins in $[f_{\min}, f_{\max}]$ (paper) or `nseg` bins in $[1/T, f_{\rm Nyq}]$ (code)?
* **Q7.** Did the code behind Pozzoli et al. contain F1 and F2?
* **Q8.** Are the cross-term sign (C2) and the π/12 arm phase (C3) intended?

## Conventions, not errors

* **C1.** The zero-width limit returns the point-source power × $\cos b_M$ (sky measure $\cos b\,db\,dl$).
* **C2.** The printed groups and the code use the opposite sign of the cross-covariance from the paper's stated $p(b, l)$, i.e. δ → −δ.
* **C3.** The paper's arm vectors give the phase $4(l + \pi/4)$; the groups and the code use $4(l + \pi/12)$, a 30° offset in β₀.
* **C4.** `cosPsi = +sqrt(1 - sinPsi**2)` covers only cos 2δ ≥ 0, so half the ellipse orientations are out of reach.

## Approximations still to check

* The model is evaluated at the coarse-bin midpoint ([setting_data.py:91](../bahamas/method/setting_data.py#L91)), not averaged over the bin.
* Within-chunk modulation is ignored (up to a factor ≈2 in $A^2$ within some monthly chunks).
* With 3 h unscheduled gaps every ~10 days, only about 100 days per year survive the 0.9 × `sched_period` rule.

## Reproduce

```
cd simulations/30_days/needlet_patches
../../.venv/bin/python paper_terms_check.py     # F1, F2, M1-M3 term by term
../../.venv/bin/python envelope_check.py        # first-principles envelope comparison
cd ../../.. && .venv/bin/python -m pytest tests/test_paper_bugs.py
```
