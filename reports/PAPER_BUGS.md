# Galactic envelope: code errors (fixed) and misprints in the paper

The analytic envelope of the Galactic foreground in bahamas, `envelopes_gaussian`
([modulation.py](../bahamas/psd_response/modulation.py)) and its chunk average `average_envelopes_gaussian`
([average_envelope.py](../bahamas/psd_response/average_envelope.py)), implements App. A of
Buscicchio et al. 2024, *A test for LISA foreground Gaussianity and stationarity. I. Galactic white-dwarf binaries*
([arXiv:2410.08263](https://arxiv.org/abs/2410.08263)). We checked every term of the code against the paper
(Oct 12, 2026).

**Summary**

| | where | status |
|---|---|---|
| a `cosh` that should be `sinh` | code, H_S;(3) | **fixed** |
| missing $e^{-2\sigma_+^2}$, $e^{-6\sigma_+^2}$ on 48 terms | code, all Δφ groups | **fixed** |
| −240 should be −2430 | paper, H_D;(4) | misprint; the code is right |
| `4φ_L` should be `4Δφ_L` | paper, H_D;(4) | misprint; the code is right |
| "sin Δ3φ_L" should be sin 3Δφ_L | paper, H_S;(3) | misprint; the code is right |
| sign of the cosh/sinh cross terms; arm-labelling phase | paper | conventions, not errors |

The errors were in the code up to commit `4db1cc9`. Runs 1 and 2 ([WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md),
[WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md)) used the pre-fix code and have not been rerun.

## How it was checked

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
| paper as printed | 44 of 45 (H_D;(4) cos 3b is the misprint below) |
| code before the fix (`4db1cc9`) | 17 of 45 |
| code after the fix | 45 of 45 |

A second check, [envelope_check.py](../data/needlet_patches/envelope_check.py), compares the envelope with a
first-principles low-frequency response from the bahamas orbits. It tests a point source (with the paper's cos b sky
measure, see below) and an exact numerical Gaussian sky average:

| | point source, A and E | parity violation | Gaussian sky average (pack-2 widths, 1 yr) |
|---|---|---|---|
| before the fix | 0.9% | 1% | 5.3% |
| after the fix | 0.5% | 2×10⁻¹⁵ | 0.4% |

The remaining 0.4–0.5% comes from the orbit models: the paper uses a rigid constellation, bahamas an eccentric orbit.

## Errors in the code (fixed)

Line numbers refer to `modulation.py`; in `average_envelope.py` the same lines are 12 further down.

### 1. `cosh` instead of `sinh` in H_S;(3)

[modulation.py:156](../bahamas/psd_response/modulation.py#L156), the −864 term. The paper prints

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

### 2. Missing damping on the 3b and 5b harmonics

In every group the paper multiplies the cos 3b / sin 3b terms by $e^{-2\sigma_+^2}$ and the cos 5b / sin 5b terms by
$e^{-6\sigma_+^2}$, e.g. H_S;(1):

```math
-32112\, e^{-2\sigma_+^2}\left(\cosh\frac{3\sigma_s^2}{2}\sin 3b_M\cos\Delta\phi_L + \sinh\frac{3\sigma_s^2}{2}\cos 3b_M\sin\Delta\phi_L\right)
```

The code had these factors (`fact3b`, `fact5b`) only in H_S;(0) (line 143) and H_D;(4) (line 169). It dropped
them in the other 24 + 24 terms of `Sum` (lines 146–164) and `Diff` (lines 171–202). The fix puts them on the
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
average of the instantaneous envelope, pass with both files fixed.

## Misprints in the paper (the code is right)

All three are in App. A.2. The TeX lines refer to `main.tex` of the arXiv source.

1. **H_D;(4), the cos 3b coefficient** (TeX line 880). Printed:

   ```math
   e^{-\frac{1}{4}\sigma_S^2-\frac{1}{4}\sigma_c^2}\left(324\cos b_M - 240\, e^{-2\sigma_+^2}\cos 3b_M - 5670\, e^{-6\sigma_+^2}\cos 5b_M\right)\cos\left(4\phi_L+4\bar l_M\right)
   ```

   The Gaussian average of the paper's own $h_{D;(4)}$ gives **−2430**, not −240. The code has −2430.

2. **H_D;(4), the phase** (same line). Printed $\cos(4\phi_L + 4\bar l_M)$; $\phi_L$ on its own is not defined
   in the paper. The derivation gives $\cos(4\Delta\phi_L + 4\bar l_M)$, with $\Delta\phi_L = \Omega t - l_M$, as for every
   other H_D;(n). The code uses `c4DphiL_4phiMbar`, i.e. $\cos(4\Delta\phi_L + 4\bar l_M)$.

3. **H_S;(3), the 3b and 5b terms** (TeX lines 869–870). Both print "sin Δ3φ_L" where they mean sin 3Δφ_L.

## Conventions, not errors

These do not make any result wrong, but they matter when the parameters are read physically.

* **Point-source limit includes cos b.** The paper normalises the Gaussian in the (b, l) plane and integrates
  with the sky measure cos b db dl. At zero width the formula therefore returns the point-source power times
  cos b_M, not the power itself.
* **Sign of the cross terms.** The paper's distribution p(b, l) has covariance
  $\Sigma_{bl} = -\tfrac{1}{2}\sigma_D^2\sin 2\delta$, consistent with its printed characteristic function
  (the $+\tfrac{mn}{2}\sigma_D^2\sin 2\delta$ term). The printed groups, and the code, follow the opposite sign:
  they describe the stated Gaussian rotated by −δ instead of δ.
* **Arm labelling.** With the arm vectors $\hat L_1, \hat L_2, \hat L_3$ exactly as printed (α₀ = β₀ = 0), the
  detector gives the h_D phase $4(l + \pi/4)$, not $4\bar l = 4(l + \pi/12)$. This is a 30° offset in β₀. The groups
  and the code use π/12, and the code's A and E match the bahamas-orbit response individually.
* **Orientation range in the code.** The code takes `sinPsi` = sin 2δ and sets `cosPsi = +sqrt(1 - sinPsi**2)`, so
  only cos 2δ ≥ 0 can be represented. With s1 < s2 fixed by the priors, half of the ellipse orientations are out of
  reach.

## Impact

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
* **Runs 1 and 2** used the pre-fix code for simulation and for both inferences (wavelet and OG). They are internally
  consistent and their conclusions stand, but run 2's injected "true sky envelope" is not the envelope of a Gaussian
  Galaxy with the stated parameters.
* **Paper 2** ([arXiv:2511.03604](https://arxiv.org/abs/2511.03604)) uses bahamas and generates its data from the same
  model it fits, so its injection/recovery results are self-consistent in the same way. Its comparison with the
  independent Yorsh 1b simulation, and any physical reading of σ₁², σ₂² and ψ, would be affected if its code had these
  errors. We have only checked the code in this repository.

## Reproduce

```
cd data/needlet_patches
../../.venv/bin/python paper_terms_check.py   # term-by-term table (paper, code before the fix, code now), ~1 min
../../.venv/bin/python envelope_check.py      # first-principles comparison and impact on the pack-2 envelope
```

`paper_terms_check.py` reads the pre-fix code from git (commit `4db1cc9`).
