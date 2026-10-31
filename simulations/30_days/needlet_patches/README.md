# Bright patches on the Galactic foreground: what can a needlet sky model recover? (30 days)

A separate study from runs 1 and 2 of [simulations/30_days/](../). We simulate a Galaxy with a
few compact patches of **excess power** and ask whether a needlet description of the sky can find them again in
30 days of LISA A/E data. A and E are treated as uncorrelated, as in the wavelet pipeline.

| stage | question | status | script | outputs |
|---|---|---|---|---|
| 0 | forward model sky map → $P_c(t)$, and checks of it | **done** (Oct 12) | [stage0_forward.py](stage0_forward.py) | `outputs/stage0/` |
| 1 | what the data can see of the sky: Fisher eigenmodes, needlet bands, patch detection and localisation, needlet reconstruction from noise-free data | **done** (Oct 12) | [stage1_identifiability.py](stage1_identifiability.py) | `outputs/stage1/` |
| 2 | needlet fit of the noisy simulated data (sampler) | **done** (Oct 18) as run 5c, on run 5's sky (real patch positions), see [Run 5c](#run-5c-the-excess-in-needlets-result) | [../run5/needlet_fit.py](../run5/needlet_fit.py) | `../outputs/run5/5c/` |

**Short answer so far.** Over 30 days the data constrain about **5 sky patterns**, all with ℓ ≤ 4. Patches carrying 25%
of the Galaxy's flux each are detected at 5–10σ, but **only if the Galaxy's shape is known**; with a free disk/bulge
shape this falls to 0.2–3σ. Even then a patch cannot be localised: any position at the same angle from the
constellation normal fits almost equally well. A needlet basis does not get around this. Only needlets with ℓ ≤ 4
content (bands j = −1, 1, 2) receive any information; j = 0 and j ≥ 3 are blind. A needlet fit of noisy data with the
spectra free ([run 5c](#run-5c-the-excess-in-needlets-result), Oct 18) recovers the total galactic modulation as well as
any other model, but only 3 sky directions are narrowed, the excess's amount trades against the Galaxy's
amplitude and is set by the prior, and nothing is located.

## How the excess power is simulated

All Galactic power (smooth Galaxy and patches) comes from unresolved DWDs with **one common spectrum**
$S_{\rm gal}(f)$. The components differ only in where the power sits on the sky. The sky is a HEALPix map
$I(\hat n)$ of **flux per pixel** (nside 32, 1.8° pixels, ecliptic coordinates like the orbit). The time series is
generated exactly as in runs 1 and 2:

```math
x_c(t) = n_c(t) + \sqrt{P_c(t)}\, g_c(t), \qquad S_c(f,t) = S_n(f) + P_c(t)\, S_{\rm gal}(f),
```

with $g_c$ stationary Gaussian with PSD $S_{\rm gal}$ (`bahamas.wavelet.simulate.simulate_channels`, which accepts
any `modulation(t, channel)`). What is new is the modulation, which is **linear in the sky map**:

```math
P_c(t) = \frac{1}{\bar R} \sum_{\rm pix} I({\rm pix})\, R_c(t, {\rm pix}), \qquad R_c = F_{+,c}^2 + F_{\times,c}^2 .
```

* $R_c(t, \hat n)$ is the low-frequency power response of channel c to a point source in direction $\hat n$,
  averaged over polarisation ([sky.py](sky.py) `point_response`). It is computed from first principles: the arm unit
  vectors come from the spacecraft positions of `bahamas.psd_response.orbits.SpacecraftOrbit`; the Michelson X, Y, Z
  are half the difference of the two arm responses $\hat r\cdot e^{+,\times}\cdot\hat r$ at each vertex; then
  A = (Z − X)/√2 and E = (X − 2Y + Z)/√6.
* $\bar R$ is the all-sky mean of $R_A$ (equal to that of $R_E$; constant in time to within 1%). **An isotropic sky of
  unit flux therefore has $P_A = P_E = 1$**, and $S_{\rm gal}$ is the PSD the sources would give if they were spread
  isotropically.
* **Smooth Galaxy** ([config/sky.yaml](config/sky.yaml)): exponential disk ($R_d$ = 2.5 kpc, $z_d$ = 0.3 kpc) plus
  Gaussian bulge ($r_b$ = 0.5 kpc, 30% of the mass), seen from the Sun at 8.12 kpc. Pixel flux is the
  line-of-sight integral of the density, $\int\rho\thinspace ds$ (number of sources per solid angle ∝ ρ s², flux of each
  ∝ 1/s²). Normalised to total flux 1; 73% of it lies within 10° of the plane.
* **Patches**: Gaussian discs on the sphere, σ = 4°, each with **25% of the Galaxy's flux**:

  | patch | galactic (l, b) | why there |
  |---|---|---|
  | `plane` | (60°, 0°) | in the disk, away from the centre |
  | `high_lat` | (−100°, +45°) | far from the plane |
  | `anticentre` | (180°, +5°) | near the antipode of the Galactic centre: the response is parity-even, so LISA cannot tell it from the same excess towards the centre |

* **Amplitude**: the pack-2 spectrum ([config/sources.yaml](config/sources.yaml)), with `amp` raised by 0.686 dex so
  that the smooth Galaxy alone has the same 30-day mean A + E galactic power as run 2. The patches come on top
  (+75% flux).
* **Data settings**: identical to run 2. 30.00 d at Δt = 10 s, TDI gen 1, seed 2026, 2 extra days simulated at each
  end and cut; WDM with $N_f$ = 3600 (72 pixels of 10 h), 2 edge pixels dropped at each end, 2-pixel time groups
  (34 groups of 20 h), 200 log-frequency bins over 0.1–29 mHz, Gamma likelihood ([patches.py](patches.py)).

### Why not `envelopes_gaussian`, and two errors in it (fixed)

The analytic envelope of runs 1 and 2 (`bahamas/psd_response/modulation.py`, from Buscicchio et al. 2024,
[arXiv:2410.08263](https://arxiv.org/abs/2410.08263), App. A) describes one Gaussian in ecliptic (β, λ). It cannot
represent a disk, a bulge or patches, so this study builds the sky pixel by pixel instead. While checking it against
the first-principles response ([envelope_check.py](envelope_check.py), `outputs/envelope_check/`) we found the
following.

* **Convention, not an error.** The paper normalises the Gaussian in the (β, λ) plane and integrates with the sky
  measure cos β dβ dλ, so the formula's point-source limit is $R_c \cos\beta$. With that factor the point-source
  response agrees with first principles to 0.9% (0.5% after the fixes below). An earlier version of this README
  reported a 37% mismatch; that came from leaving out the cos β.

Term-by-term check against the paper's TeX source ([paper_terms_check.py](paper_terms_check.py)): all 45 printed
terms (5 H_S and 9 H_D groups × the cos b, cos 3b, cos 5b harmonics) are compared with an independent Gaussian
average of the paper's own point-source coefficients `h_S;(n)`, `h_D;(n)` (which in turn match the paper's detector
model to 10⁻¹⁴), and the code is compared statement by statement.

* **Error 1 (code, fixed Oct 12): one `cosh` should be `sinh`.** [modulation.py:156](../../../bahamas/psd_response/modulation.py#L156)
  (and [average_envelope.py:168](../../../bahamas/psd_response/average_envelope.py#L168)), the −864 term of H_S;(3):
  the code had `cosh3HalvesSigmaSqSin * s3DphiL * cbM`, the paper sinh. At zero width this leaves a parity-odd term
  (1%); with sinh the parity violation is 2×10⁻¹⁵.
* **Error 2 (code, fixed Oct 12): missing damping factors.** The paper multiplies every cos 3b / sin 3b term by $e^{-2\sigma_+^2}$
  and every cos 5b / sin 5b term by $e^{-6\sigma_+^2}$. The code had them only in H_S;(0) (line 143) and H_D;(4)
  (line 169) and dropped them in the 48 other 3b/5b terms of `Sum` (lines 146–164) and `Diff` (lines 171–202).
* **Fix** (both files; the errors were present up to commit 4db1cc9): sinh in that term, and `fact3b` / `fact5b` on
  the coefficient of every 3b / 5b term, as in the paper. The fixed code matches the derivation in all 45 terms to
  10⁻¹⁴: there is no other coefficient, sign or hyperbolic factor error. `tests/test_average_envelope.py` now pins
  2.56285 (was 2.08448), which the independent derivation reproduces to 10⁻¹⁰.
* **Misprints in the paper (the code is right):** H_D;(4) prints `−240 e^{−2σ₊²} cos 3b_M` where the derivation and
  the code give −2430; its phase prints cos(4φ_L + 4l̄_M) for cos(4Δφ_L + 4l̄_M); H_S;(3) prints "sin Δ3φ_L" for
  sin 3Δφ_L.
* **Conventions, not errors.** (i) The printed groups use the opposite sign of the cross term to the paper's
  p(b, l) and characteristic function, i.e. they (and the code) describe the stated Gaussian with δ → −δ. (ii) With
  the arm vectors as printed, the h_D phase is 4(l + π/4), not 4(l + π/12): a 30° arm-labelling (β₀) convention; the
  groups and the code use π/12, and the code's A and E match the bahamas-orbit response individually. (iii) The code
  takes sin ψ = sin 2δ and sets cos ψ = +√(1 − sin²ψ), so only orientations with cos 2δ ≥ 0 are reachable; with
  s1 < s2 fixed by the priors, half of the ellipse orientations cannot be represented.

The fixed formula reproduces the exact Gaussian sky average to 4×10⁻³ (the level set by the two orbit models); the
pre-fix code differed by 5%. `average_envelope.py` (the chunk average used by the OG pipeline) had the same two
errors and is fixed the same way.

**Consequences for runs 1 and 2 and for paper 2** (pack-2 values, 30 days):

* The pre-fix envelope differs from the fixed one by 15% in the mean of $P_A$ (3% in $P_E$), and in shape by up to 19%
  (A) and 9% (E).
* Runs 1 and 2 used the pre-fix code for both simulation and inference (wavelet and OG), so they are internally consistent
  and their conclusions (wavelet recovery; wavelet vs OG posterior widths) stand. What changes is the meaning of the
  injection. Run 2's "true sky envelope" is not the envelope of a Gaussian Galaxy with those parameters.
* The pre-fix model can mimic the correct envelope to 0.7% with shifted parameters: s1 0.042 → 0.100 (the prior
  edge), s2 0.138 → 0.260, sin ψ −0.83 → −0.95, lat and long within 0.4σ, amplitude +0.04 dex. Relative to the
  run-2 OG posterior widths these are shifts of 2.2σ (s1), 2.2σ (s2) and 0.9σ (sin ψ). A fit of correctly modelled
  (or real) data with the code would therefore overestimate the Galaxy's widths.
* Paper 2 ([arXiv:2511.03604](https://arxiv.org/abs/2511.03604)) uses bahamas and generates its data from the same
  PSD model it fits, so its injection/recovery results are self-consistent in the same way. Its comparison with the
  independent Yorsh 1b simulation, and any physical reading of σ1², σ2² and ψ, would be affected if its code has these
  errors. We have only checked the code in this repository.

Runs 1 and 2 have not been rerun with the fixed code.

## Needlets

[needlets.py](needlets.py): standard needlets (Narcowich, Petrushev & Ward 2006; Marinucci et al. 2008), B = 2:

```math
\psi_{jk}(\hat n) = \sqrt{\lambda_{jk}} \sum_\ell b\!\left(\frac{\ell}{B^j}\right) \frac{2\ell+1}{4\pi}\, P_\ell(\hat\xi_{jk}\cdot\hat n), \qquad I = \sum_{jk} \beta_{jk}\, \psi_{jk} .
```

The window $b$ is the usual smooth $C^\infty$ one. Band j covers B^(j−1) < ℓ < B^(j+1):

| band j | ℓ | cubature points ξ_jk (HEALPix nside_j = 2^j) |
|---|---|---|
| −1 (scaling) | 0 | 1 |
| 0 | 1 | 12 |
| 1 | 2, 3 | 48 |
| 2 | 3–7 (ℓ = 4 at full weight) | 192 |
| 3 | 5–15 | 768 |

The windows sum to one (tight frame). With this coarse cubature a band-limited map is reconstructed to 3.7%
(0.6% with twice the nside_j, at four times the coefficients). That is enough for a parametrisation.

**Reading the formula.** From the inside out:

* $\frac{2\ell+1}{4\pi} P_\ell(\hat\xi\cdot\hat n)$ is all of multipole ℓ arranged symmetrically around the direction
  $\hat\xi$. By the addition theorem it equals $`\sum_m Y_{\ell m}(\hat\xi)\thinspace Y^*_{\ell m}(\hat n)`$. Summed over
  every ℓ with weight one, it is a delta function at $\hat\xi$.
* The window $b(\ell/B^j)$ keeps only the band $B^{j-1} \lt \ell \lt B^{j+1}$, so the sum becomes a smooth bump about
  180°/B^j wide, built from that band alone. This is the needlet: local on the sky **and** in ℓ, as a WDM pixel is
  local in time and in frequency.
* j is the scale, k the position. The centres $\hat\xi_{jk}$ are the pixel centres of a HEALPix grid with nside 2^j,
  and $\lambda_{jk} = 4\pi/N_j$ is the area each covers.
* The coefficients $\beta_{jk}$ of $`I = \sum_{jk} \beta_{jk}\thinspace\psi_{jk}`$ are what a fit would sample.

### What the data can see of the sky (derivation)

**1. Power is the sky weighted by the antenna pattern.** From the forward model above,

```math
P_c(t) = \frac{1}{\bar R}\int I(\hat n)\, R_c(t,\hat n)\, d\Omega, \qquad R_c = F_{+,c}^2 + F_{\times,c}^2 .
```

**2. The response contains only ℓ = 0, 2, 4.** At low frequency each channel is a fixed linear combination of arm
responses, so $`F_{A,c} = D^{(c)}_{ij}\thinspace e^A_{ij}(\hat n)`$ with a detector tensor $D^{(c)}$ that does not depend
on $\hat n$ (it changes only with the constellation's orientation, i.e. with t). The polarisation tensors
$e^+ = \hat p\hat p - \hat q\hat q$ and $e^\times = \hat p\hat q + \hat q\hat p$ depend on the polarisation angle, but
their sum does not:

```math
\sum_{A=+,\times} e^A_{ij}\, e^A_{kl} = \Pi_{ik}\Pi_{jl} + \Pi_{il}\Pi_{jk} - \Pi_{ij}\Pi_{kl}, \qquad \Pi_{ij} = \delta_{ij} - n_i n_j ,
```

so

```math
R_c(t,\hat n) = D^{(c)}_{ij}(t)\, D^{(c)}_{kl}(t)\,\big(\Pi_{ik}\Pi_{jl} + \Pi_{il}\Pi_{jk} - \Pi_{ij}\Pi_{kl}\big) .
```

Each $\Pi$ has terms of order 0 and 2 in $\hat n$, so $R_c$ is a polynomial of degree 4 in the components of $\hat n$
with only even powers. On the sphere a polynomial of degree d contains multipoles ℓ = d, d − 2, …, and an even one
contains only even ℓ. Hence ℓ ∈ {0, 2, 4}. Numerically ([sky.py](sky.py) `point_response`, stage 0): $R_A$ is 92%
ℓ = 0, 7.5% ℓ = 2 and 0.2% ℓ = 4, with less than 10⁻⁸ of the power anywhere else. Evenness is also why a patch at
the anticentre looks the same as one towards the centre.

**3. The sky enters through 15 numbers.** Expand both in real spherical harmonics. By orthonormality the integral
becomes a sum, and only the multipoles present in $R_c$ survive:

```math
P_c(t) = \frac{1}{\bar R}\sum_{\ell\in\{0,2,4\}}\ \sum_{m=-\ell}^{\ell} I_{\ell m}\, R_{c,\ell m}(t) .
```

The sky is seen only through its 1 + 5 + 9 = **15** coefficients $I_{\ell m}$ with ℓ = 0, 2, 4. This is a hard
limit that holds for any amount of data: two skies with the same 15 coefficients give the same data, and any sky
built from other multipoles gives none.

**4. Thirty days measure about 5 of them.** The coefficients $R_{c,\ell m}(t)$ change only as the constellation
turns, and over 30 days its normal turns about 30°. Each $P_c(t)$ is then close to a quadratic in t (stage 0),
about 3 numbers per channel, about 6 for A and E. Stage 1 makes this precise with the Fisher matrix of the 15
$I_{\ell m}$ ([Stage 1](#stage-1-identifiability-outputsstage1-fisherjson), item 1): the Galaxy's SNR in its eigenmodes is 82,
10.3, 2.2, 2.2, 2.3, then 0.5, 0.03 and ~0, so about 5 sky patterns are measured. "About 5" counts modes with
SNR ≳ 2. It holds the spectra at their true values, so it is optimistic. A full year turns the pattern through
360°, so more of the 15 should become measurable.

**5. Which needlets matter.** Substitute $`I = \sum \beta_{jk}\psi_{jk}`$. Each needlet's harmonic coefficients are
$`\psi_{jk,\ell m} = \sqrt{\lambda_{jk}}\, b(\ell/B^j)\, Y_{\ell m}(\hat\xi_{jk})`$, so by step 3

```math
P_c(t) = \sum_{jk} \beta_{jk}\, T_{c,jk}(t), \qquad T_{c,jk}(t) = \frac{\sqrt{\lambda_{jk}}}{\bar R}\sum_{\ell\in\{0,2,4\}} b(\ell/B^j) \sum_m Y_{\ell m}(\hat\xi_{jk})\, R_{c,\ell m}(t) .
```

$T_{c,jk}$ vanishes unless the band's window covers ℓ = 0, 2 or 4: j = −1 (ℓ = 0), j = 1 (ℓ = 2, 3) and j = 2
(ℓ = 3–7) do; j = 0 (ℓ = 1 only) and j ≥ 3 (ℓ ≥ 5) do not. That leaves 1 + 48 + 192 = **241** needlet coefficients,
which still reach the data only through the same 15 combinations, of which about 5 are measured. The needlet Fisher
matrix confirms this: 15 non-zero eigenvalues, falling by 10⁶ over the first 7 (stage 1, item 2).

### Templates and needlets: one linear model

Because $P_c(t)$ is linear in the sky map, any sky written as a sum of pieces, $`I = \sum_a \theta_a\thinspace p_a`$,
gives

```math
P_c(t) = \sum_a \theta_a\, T_{c,a}(t), \qquad T_{c,a}(t) = \frac{1}{\bar R}\int p_a(\hat n)\, R_c(t,\hat n)\, d\Omega ,
```

where $T_{c,a}(t)$ is the modulation that piece a would produce on its own. Run 5a and the planned needlet fit are
both of this form and differ only in the pieces:

| model | pieces $p_a$ | coefficients $\theta_a$ | free | status |
|---|---|---|---|---|
| run 5a ([patch_fit.py](../run5/patch_fit.py)) | smooth Galaxy (shape at truth, coefficient 1) and each patch with unit flux at its true position and shape | $F_i$, the patch's flux in units of the Galaxy's | 4 | done |
| needlets (5c, [needlet_fit.py](../run5/needlet_fit.py)) | $\psi_{jk}$ on the HEALPix grids of bands j = −1, 1, 2 | $\beta_{jk}$ | 241 | done, [below](#run-5c-the-excess-in-needlets-result) |

In run 5a, $T_{c,i}(t)$ is computed once from the known patch and only the $F_i$ are sampled:
$`P_c(t) = T_{c,\mathrm{gal}}(t) + \sum_i F_i\thinspace T_{c,i}(t)`$. A needlet fit assumes no positions, but by step 3
any choice of pieces reaches the data only through the 15 $I_{\ell m}$. So the patch fluxes of 5a, where the
positions are given, are the best a needlet fit could do. The wavelet atoms of runs 1–4 are not of this form: they
model $\ln P_c(t)$ directly, separately for each channel, and say nothing about the sky.

### Run 5c: the excess in needlets (result)

Run 5's data (run 2's smooth Galaxy as a sky map plus LMC, SMC, Sgr dSph and the near end of the bar with 20%, 8%,
12% and 12% of its flux; same noise as run 2; [simulations/30_days/](../)), same WDM Gamma
likelihood as runs 1–5a. Model: the smooth Galaxy fixed at the truth (shape and unit flux, as in 5a) plus the excess
as needlets of bands j = −1, 1, 2 on the response grid (nside 32), with `amp` and the six other spectral parameters
free. Prior $`\beta_{jk} \sim N(0, s_j^2)`$, $s_j$ = rms of the smooth Galaxy's own needlet coefficients in band j
(as in stage 1, item 5), so nothing about the patches. Code: [needlet_fit.py](../run5/needlet_fit.py);
analysis: [run5c_needlets.ipynb](../run5/run5c_needlets.ipynb).

**Sampling, exact at low dimension.** With $z = \beta/s \sim N(0, I)$ the modulation is $T_{\mathrm{gal}} + Mz$ with
$M = T\thinspace\mathrm{diag}(s)$. On the 34 time blocks of 30 days, $M$ has 11 singular values above $10^{-6}$ of
the largest (the next is $3\times10^{-7}$), so the likelihood sees only $y = V_{11}^T z$. NUTS samples y and the 7
spectral parameters; the other 230 directions of z do not enter the likelihood and are drawn from their prior. That
is the exact posterior of the 241-coefficient model. In $(\mathrm{amp}, y)$ the two trade along a curved ridge
(correlation 0.985) and a first run had 7% divergent transitions; sampling $q = 10^{\mathrm{amp}-\mathrm{amp}_0}\thinspace y$
makes the ridge straight. Final run: 4 × (1000 + 2000), target acceptance 0.95, 10 divergences (0.1%),
$\hat R \le 1.002$, $n_{\mathrm{eff}} \ge 1900$, 34 min. A second run with every $s_j$ ×3 tests the prior.

| | atoms (run 5) | 5a: known positions | 5c: needlets | 5c, prior ×3 |
|---|---|---|---|---|
| 30-day brightness A, recovered / injected | 1.015 | 1.014 | 1.011 | |
| 30-day brightness E | 0.990 | 0.988 | 0.992 | |
| 90% width | 3.2% | 3.2% | 3.3% | |
| total excess flux (truth 0.52) | – | 0.68 ± 0.12 | 0.17 ± 0.21 | 0.73 ± 0.56 |
| amp − true | – | −0.05 dex | +0.12 dex | −0.03 dex |
| visible directions narrowed below ½ the prior | – | – | 3 of 11 | 3 of 11 |
| corr(posterior mean, true excess), ℓ = 0, 2, 4 parts | – | – | 0.44 | 0.69 |
| max \|mean / sd\| on the sky | – | – | 1.1 | 0.9 |

* **Total power: as good as the other models.** All three recover the 30-day galactic brightness to about 1% with
  the same 3.2–3.4% width, and the truth is inside 5c's 90% band at every analysed time. The spectral parameters
  agree across the three.
* **How much excess: set by the prior.** The posterior total excess flux moves from 0.17 to 0.73 when the prior
  widens. The Galaxy's amplitude takes up the difference: in 5c amp comes out 0.12 dex (×1.33) high,
  anticorrelated with the excess flux (−0.72). With the Galaxy's shape known but its amplitude free, an excess
  spread over needlets has almost the Galaxy's own time profile, so the data fix the sum and the prior sets the
  split. 5a avoids this because the patch positions, and so their distinct time profiles, are given.
* **How many sky directions: 3, and not by the data alone.** Of the 11 visible directions, 3 have a posterior sd
  below half the prior (0.11, 0.23, 0.22), 2 are narrowed a little (0.80, 0.76), 6 not at all. Tripling the prior
  widens even the best three 1.9–2.4× (data-dominated would be 1×, pure prior 3×), because they trade against amp
  and the spectral parameters. This is lower than the "about 5" of stage 1's Fisher count (item 1), which held the
  spectra at the truth and had no noise.
* **Where: not located.** The posterior mean has some of the true excess's visible structure (correlation 0.44),
  but nowhere exceeds 1.1σ, as the noise-free Wiener filter of stage 1 found (correlation 0.50, max 0.5–1.4).
* **The prior is too narrow for compact patches.** Scaled to the smooth Galaxy, it puts the true excess at up to
  3.5σ in single directions (compact patches carry more ℓ = 2, 4 power than a smooth Galaxy). Unmeasured directions
  shrink to zero; one ends 3.4σ from the truth for that reason alone.

**Conclusion.** In 30 days a sky model without positions recovers the total galactic modulation as well as the
atoms or the known-position templates, but it can neither separate an excess from the Galaxy's brightness nor
place it. What 5a measures about the patches comes from knowing where they are. The year-long Fisher calculation
under [Next](#next) is the cheapest way to see whether 360° of response rotation changes this.

## Stage 0: forward model (`outputs/stage0/`)

* `response_checks.png`, `checks.json`: closed-form match 6×10⁻⁴; parity exact (10⁻¹⁵); harmonic content of
  $R_A$: ℓ = 0 (92%), 2 (7.5%), 4 (0.2%); the isotropic sky gives P between 0.998 and 1.007 over a year.
* `sky.png`: the sky, and what of it survives in ℓ = 0, 2, 4. The visible Galaxy is a blob at the centre **and** at
  the anticentre (parity). Each patch keeps 26% of its map norm, spread into ℓ = 2/4 lobes tens of degrees wide.
* `modulation.png`: over 30 days each $P_c(t)$ is close to a quadratic. The patches add 10–75% to the galactic
  power, with time dependences of their own. For example, the `high_lat` excess in A falls from +76% to +26%
  relative to the Galaxy.

## Stage 1: identifiability (`outputs/stage1/`, `fisher.json`)

Everything is linear in the map, so for any sky parametrisation the Fisher matrix is $J^T W J$. Here
$`W_c[n] = \sum_g \tfrac{\nu_g}{2} \big(S_{{\rm gal},g}/(P_c[n] S_{{\rm gal},g} + S_{n,cg})\big)^2`$ is the information
on $P_c$ in time group n (Gamma likelihood, true spectra), and J is the response of $P_c$ to the parameters. Each
20-h group measures $P_c$ to ±0.05 (about 6%). The spectra are held at their true values (see caveats below).

**1. Visible sky patterns** (`modes.png`). Fisher eigenmodes of the 15 even ℓ ≤ 4 harmonics, best first:

| mode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | … 15 |
|---|---|---|---|---|---|---|---|---|
| σ (intensity coefficient) | 0.0015 | 0.0135 | 0.0139 | 0.043 | 0.13 | 0.27 | 1.9 | 2×10⁵ |
| Galaxy SNR in the mode | 82 | 10.3 | 2.2 | 2.2 | 2.3 | 0.5 | 0.03 | ~0 |

Mode 1 is essentially the response-weighted total flux. About 5 patterns are measured; the rest are set by the
prior.

**2. Needlet bands** (`needlet_bands.png`). Share of the Fisher information: j = −1 70%, j = 1 29%, j = 2 1.4%,
j = 0 ~10⁻³², j = 3 ~10⁻⁷ (pixelisation leakage). The needlet Fisher matrix (K = 1021) has 15 non-zero eigenvalues,
and they fall by 10⁶ over the first 7.

**3. Patch detection.** SNR of each patch's contribution to $P_c(t)$:

| patch (25% flux) | Galaxy known | Galaxy flux free | Galaxy flux + shape free ($R_d$, $z_d$, $r_b$, bulge fraction) | flux for 5σ, flux free / flux + shape free |
|---|---|---|---|---|
| `plane` | 14.7 | 5.8 | 3.1 | 0.22 / 0.41 |
| `high_lat` | 43.7 | 10.3 | 1.3 | 0.12 / 0.93 |
| `anticentre` | 14.9 | 5.0 | 0.17 | 0.25 / 7.6 |

The SNR scales linearly with patch flux. The anticentre patch is absorbed almost completely by a slightly
different Galaxy (parity: it is equivalent to more bulge).

**4. Localisation** (`localisation.png`, `chi2_position_maps.npz`). Δχ² of moving one patch anywhere on the sky,
re-fitting its flux and the Galaxy flux, from noise-free data. The allowed region is **not a blob but a ring**: the
positions at the same angle from the mean constellation normal. For the Δχ² < 1 positions, the spread of
|cos(angle to the normal)| is 0.05, against 0.29 over the whole sky, and its mean matches the patch's own value
(0.15 vs 0.15, 0.60 vs 0.63, 0.09 vs 0.13). Reason: in the low-frequency limit $R_A + R_E$ depends only on
$\hat n\cdot\hat z(t)$, and $\hat z$ turns only ~30° in 30 days, so only the small A − E difference breaks the
symmetry. Moving a patch to the worst position costs only 5–10σ in total. Local Fisher errors (a few degrees) are
therefore misleading. With the Galaxy shape and the other patches free, the Fisher matrix for (flux, l, b) is
singular.

**5. Needlet reconstruction** (`reconstruction.png`). Wiener filter of the excess in needlets j ≤ 2 (K = 253) from
noise-free data, with the Galaxy nuisances free (linearised) and $\beta_{jk} \sim N(0, s_j^2)$, where $s_j$ is the
rms of the Galaxy's own needlet coefficients in band j (no knowledge of the patches). The posterior mean correlates
0.50 with the visible (ℓ = 0, 2, 4) part of the true excess, but nowhere is it significant: max |mean/std| = 0.5
(1.4 with the Galaxy shape fixed). The excess is spread over broad ℓ = 2 lobes, as expected.

### Caveats

* Spectra (noise and $S_{\rm gal}$ shape) are fixed at their true values in stage 1. Freeing them can only lower
  the numbers above.
* The orbit's start phase is t0 = 0 of `SpacecraftOrbit`, as in runs 1 and 2. Which ring a patch falls on, and how
  well the A − E difference breaks it, depends on the season.
* The Fisher results are local and Gaussian. The ring scan (item 4) is the exception: it is a full scan of
  noise-free data.

## Next

Stage 1 and run 5c answer the question for 30 days: the total modulation is measured, the excess's amount and
position are not. Options, cheapest first:

1. **Fisher for a full year.** No time series is needed: only the band-averaged spectra and the response at the
   time groups, so it costs minutes on a laptop. This would show how the ring degeneracy breaks as $\hat z$
   precesses through 360°, and how many of the 15 visible numbers a year measures.
2. **Run 5c on a year of data**, if the Fisher count says it is worth it. The sampler already reduces the needlets to
   their data-visible directions exactly, so the cost grows with the number of time blocks, not of needlets.
3. **A prior matched to compact sources**, e.g. $s_j$ from a population of patches rather than the smooth Galaxy:
   in 5c the Galaxy-scaled prior shrinks the true excess.
4. Brighter or more numerous patches: the SNR is linear in flux, so the table above gives the thresholds directly.

## Files

```
config/sky.yaml          Galaxy (disk + bulge) and patch definitions, map resolution
config/sources.yaml      noise and galactic spectrum (pack-2 values; amp is offset in patches.py)
sky.py                   coordinates, first-principles A/E point response, SkyResponse (P = R @ I), Galaxy and
                         patch maps, band-limiting to l = 0, 2, 4
needlets.py              needlet windows and NeedletFrame (analysis / synthesis on HEALPix)
patches.py               Truth (maps + response + amplitude), simulate(), likelihood(), spectra(), fisher_P()
plotting.py              shared figure style
envelope_check.py        python envelope_check.py         -> outputs/envelope_check/  (envelopes_gaussian vs first principles)
paper_terms_check.py     python paper_terms_check.py      (term-by-term check against arXiv:2410.08263 App. A)
stage0_forward.py        python stage0_forward.py         -> outputs/stage0/
stage1_identifiability.py python stage1_identifiability.py -> outputs/stage1/   (about 2 min, 8 cores)
outputs/
  envelope_check/        envelope_check.json, envelope_check.png
  sim/data.h5            the simulated 30-day WDM data (seed 2026)
  stage0/                checks.json, response_checks.png, sky.png, modulation.png
  stage1/                fisher.json, modes.png, needlet_bands.png, localisation.png, chi2_position_maps.npz,
                         reconstruction.png
```

Run from this folder with the project venv: `../../.venv/bin/python stage0_forward.py`.
