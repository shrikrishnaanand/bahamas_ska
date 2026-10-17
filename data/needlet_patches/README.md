# Bright patches on the Galactic foreground: what can a needlet sky model recover? (30 days)

A separate study from runs 1 and 2 of [data/wavelet_injection/](../wavelet_injection/). We simulate a Galaxy with a
few compact patches of **excess power** and ask whether a needlet description of the sky can find them again in
30 days of LISA A/E data. A and E are treated as uncorrelated, as in the wavelet pipeline.

| stage | question | status | script | outputs |
|---|---|---|---|---|
| 0 | forward model sky map → $P_c(t)$, and checks of it | **done** (Oct 12) | [stage0_forward.py](stage0_forward.py) | `outputs/stage0/` |
| 1 | what the data can see of the sky: Fisher eigenmodes, needlet bands, patch detection and localisation, needlet reconstruction from noise-free data | **done** (Oct 12) | [stage1_identifiability.py](stage1_identifiability.py) | `outputs/stage1/` |
| 2 | needlet fit of the noisy simulated data (sampler) | not started, see [Next](#next) | | |

**Short answer so far.** Over 30 days the data constrain about **5 sky patterns**, all with ℓ ≤ 4. Patches carrying 25%
of the Galaxy's flux each are detected at 5–10σ, but **only if the Galaxy's shape is known**; with a free disk/bulge
shape this falls to 0.2–3σ. Even then a patch cannot be localised: any position at the same angle from the
constellation normal fits almost equally well. A needlet basis does not get around this. Only needlets with ℓ ≤ 4
content (bands j = −1, 1, 2) receive any information; j = 0 and j ≥ 3 are blind.

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

* **Error 1 (code, fixed Oct 12): one `cosh` should be `sinh`.** [modulation.py:156](../../bahamas/psd_response/modulation.py#L156)
  (and [average_envelope.py:168](../../bahamas/psd_response/average_envelope.py#L168)), the −864 term of H_S;(3):
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

**Why only a few needlets matter.** $R_c(t,\cdot)$ contains only ℓ = 0, 2, 4 (checked: power elsewhere < 10⁻⁸).
The data therefore see only the projection of the sky onto 15 real harmonics. Needlet coefficients enter
$P_c(t)$ through $\int \psi_{jk} R_c\thinspace d\Omega$, which is zero for j = 0 (ℓ = 1) and for j ≥ 3 (ℓ ≥ 5).

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

Stage 1 already answers the question for 30 days. A noisy needlet MCMC (stage 2) would reproduce it with scatter;
it would not change it. Options, cheapest first:

1. **Fisher for a full year.** No time series is needed: only the band-averaged spectra and the response at the
   time groups, so it costs minutes on a laptop. This would show how the ring degeneracy breaks as $\hat z$
   precesses through 360°.
2. **Stage 2 on the 30-day data.** Sample the excess in the Fisher eigenbasis of the visible subspace (≤ 15
   parameters plus 5 Galaxy nuisances, with eryn), and take the null space analytically from the prior. That
   demonstrates the stage-1 picture on noisy data at laptop cost.
3. Brighter or more numerous patches: the SNR is linear in flux, so the table above gives the thresholds directly.

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
