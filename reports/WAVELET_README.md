# Wavelet-domain analysis of the galactic foreground

This document describes the wavelet-domain analysis added to BAHAMAS as an **optional** alternative to the chunked
short-time Fourier transform (STFT) analysis of
[Pozzoli et al. 2025 (arXiv:2506.22542)](https://arxiv.org/abs/2506.22542) and
[Buscicchio et al. 2025 (arXiv:2511.03604)](https://arxiv.org/abs/2511.03604). Nothing in the chunked pipeline was
replaced; the wavelet pipeline runs only when a config sets `domain: 'wavelet'`.

The document gives the mathematics end to end, in one notation:

1. the data model (Section 2);
2. the WDM transform that takes the time series to the time-frequency plane (Section 3);
3. the modulation model, i.e. the "wavelets" that describe $P_c(t)$ (Section 4);
4. the likelihood (Section 5), and the proof that it is the same Gamma likelihood as Eq. 9 of arXiv:2511.03604,
   written in a different domain (Section 6);
5. how the fit is done: priors, reversible-jump MCMC and Bayes factors (Section 7).

The code, validation, configuration and limitations follow (Sections 8–13).

> **Two different "wavelets".** The word is used for two unrelated things, kept apart throughout:
>
> * the **WDM basis functions** $g_{nm}(t)$ (Section 3). They transform the *data*. They are fixed by the grid and
>   have no free parameters.
> * the **modulation atoms** $\psi\big((t-t_k)/\tau_k\big)$ (Section 4). They model the *envelope* $P_c(t)$. Their
>   number, positions, widths and amplitudes are what the RJMCMC samples. The code and configs call them
>   "wavelets" (`modulation: 'wavelets'`, `max_wavelets`); this document calls them **atoms**. The default atom is a
>   Gaussian, which strictly is not a wavelet (its mean is not zero); the optional `ricker` atom is one.

---

## 1. Notation

**Indices.**

| index | runs over | meaning |
|---|---|---|
| $c$ | $\lbrace A, E\rbrace$ | TDI channel |
| $l$ | $0, \dots, N-1$ | time sample, $t_l = t_0 + l\thinspace \Delta t$ |
| $n$ | $0, \dots, N_t-1$ | WDM time pixel, centred at $t_n = t_0 + n\thinspace \Delta T$ |
| $m$ | $0, \dots, N_f-1$ | WDM frequency pixel, centred at $f_m = m\thinspace \Delta F$ |
| $g$ | $1, \dots, N_g$ | frequency group: a set $\mathcal M_g$ of adjacent pixels $m$ (log-spaced) |
| $s$ | $1, \dots, N_s$ | time block: a set $\mathcal N_s$ of $n_\tau$ consecutive time pixels $n$ |
| $k$ | $1, \dots, K_c$ | modulation atom in channel $c$ |

**Symbols.**

| symbol | meaning | type |
|---|---|---|
| $x_c(t_l)$ | the time series of channel $c$ | data |
| $w_{c,nm}$ | WDM coefficient of $x_c$ in pixel $(n, m)$ | data |
| $Y_{c,sg}$ | $\sum w^2$ over the pixels of block $s$ and group $g$ | data (sufficient statistic) |
| $\nu_g$ | number of pixels in a group, $n_\tau\thinspace \lvert\mathcal M_g\rvert$ | fixed by the grid |
| $S_{n,c}(f)$ | one-sided PSD of the instrument noise (parameters `A`, `P`) | model, noise |
| $S_{\rm gal}(f)$ | one-sided PSD of the unmodulated galaxy (`alpha`, `amp`, `fknee`, `fr1`, `fr2`), $\propto A_{\rm gal} = 10^{\rm amp}$ | model, signal |
| $P_c(t)$ | power modulation of the galaxy in channel $c$, dimensionless | model, signal |
| $\sigma^2_{c,nm}$, $\sigma^2_{c,sg}$ | model variance of a coefficient | model (noise + signal) |
| $b_c$ | constant level of $\ln P_c$ ($b_A \equiv 0$, $b_E$ = `level_E`) | model parameter |
| $t_k, \tau_k, a_k$ | centre, width, amplitude of atom $k$ | model parameters |
| $\langle S\rangle_m$ | PSD averaged over the spectral window of pixel $m$ (Section 3.4) | derived |

Run-1/run-2 values of the grid: $\Delta t = 10$ s, $N_f = 3600$, $\Delta T = 10$ h, $\Delta F = 13.9\ \mu\mathrm{Hz}$,
$N_t = 72$ (30 days), $n_\tau = 2$, two edge pixels dropped at each end, so $N_s = 34$ blocks, and $N_g = 174$ groups
in $[0.1, 29]$ mHz.

---

## 2. Data model

For channel $c$ the data are a locally stationary Gaussian process: stationary noise plus a stationary galaxy whose
amplitude is modulated in time,

```math
x_c(t) = n_c(t) + \sqrt{P_c(t)}\; g_c(t),
```

with $n_c$ and $g_c$ independent stationary Gaussian processes with one-sided PSDs $S_{n,c}(f)$ and $S_{\rm gal}(f)$.
Its time-dependent (evolutionary) PSD is

```math
S_c(f, t) = S_{n,c}(f) + P_c(t)\, S_{\rm gal}(f).
```

$S_{n,c}$ and $S_{\rm gal}$ are the existing BAHAMAS functions `psd_noise.noise` and
`psd_galaxy.galactic_foreground`, unchanged.

**Link to the chunked model.** For the sky-envelope model, $P_c(t) = M_c^2(t)$, the square of the amplitude envelope
`psd_response.modulation.envelopes_gaussian`. The chunked analysis uses its average over a chunk
$[t_S, t_E]$,

```math
\overline M_c = \frac{1}{t_E - t_S}\int_{t_S}^{t_E} M_c^2(t)\,dt = \texttt{average\_envelopes\_gaussian}(t_S, t_E),
```

which is checked numerically to $10^{-7}$
([test_power_modulation_averages_to_chunk_envelope](../tests/test_wavelet_model.py#L52)). The two pipelines therefore
describe the same physics; the wavelet one does not average over chunks.

**Simulation.** [simulate_channels](../bahamas/wavelet/simulate.py#L144) draws $n_c$ and $g_c$ in the frequency
domain ([stationary_gaussian](../bahamas/wavelet/simulate.py#L41)), multiplies $g_c$ by $\sqrt{P_c(t)}$ (evaluated
hourly and interpolated, error $\lesssim 10^{-6}$) and adds the two. The draws are circular, so the injection runs
simulate a longer series and cut it, to avoid ends that wrap around. Random numbers use `jax.random` keys.

---

## 3. The WDM transform

**Code:** [wdm.py](../bahamas/wavelet/wdm.py). References: Necula et al. 2012; Cornish 2020
([arXiv:2009.00043](https://arxiv.org/abs/2009.00043)); Digman & Cornish 2022
([arXiv:2212.04600](https://arxiv.org/abs/2212.04600)).

### 3.1 Grid

A series of $N = N_t N_f$ samples at cadence $\Delta t$ is mapped to $N_t \times N_f$ coefficients on a uniform
time-frequency grid:

```math
\Delta T = N_f\,\Delta t, \qquad \Delta F = \frac{1}{2\,\Delta T} = \frac{1}{2 N_f\,\Delta t}, \qquad
\Delta T\,\Delta F = \tfrac12 .
```

Pixel $(n, m)$ is centred at $t_n = t_0 + n\Delta T$, $f_m = m\Delta F$
([wdm_grid](../bahamas/wavelet/wdm.py#L84)). $N_f$ and $N_t$ must be even. Each pixel covers half a unit of
time-frequency area, so each pixel holds **one real number**: $N$ samples in, $N$ coefficients out.

### 3.2 Basis functions

The WDM basis is a Wilson basis built on one window $\phi(t)$:

```math
g_{nm}(t) = \sqrt2\,\phi(t - n\Delta T)\times
\begin{cases}
\cos(2\pi f_m t), & n + m \text{ even}, \\ \sin(2\pi f_m t), & n + m \text{ odd},
\end{cases}
\qquad m = 1, \dots, N_f - 1,
```

(up to the sign convention of the odd terms, which we take from `pywavelet`). Row $m = 0$ holds the DC component on
even $n$ and the Nyquist component on odd $n$. The window is defined by its Fourier transform, the Meyer window of
Cornish 2020 Eq. 11, in angular frequency $\omega = 2\pi f$:

```math
\tilde\Phi(\omega) = \frac{1}{\sqrt{\Delta\Omega}}\times
\begin{cases}
1, & \lvert\omega\rvert < A,\\[2pt]
\cos\Big[\dfrac{\pi}{2}\,\nu_d\Big(\dfrac{\lvert\omega\rvert - A}{B}\Big)\Big], & A \le \lvert\omega\rvert < A + B,\\[2pt]
0, & \text{otherwise,}
\end{cases}
\qquad \Delta\Omega = 2\pi\Delta F,\quad A = \frac{\Delta\Omega}{4},\quad B = \frac{\Delta\Omega}{2},
```

where $\nu_d(x) = I_x(d, d)$ is the regularised incomplete beta function and $d$ sets the steepness of the roll-off
(config `nx`, default 4). Code: [phitilde](../bahamas/wavelet/wdm.py#L42). The window is flat over $\Delta F/2$ and
vanishes beyond $\lvert f - f_m\rvert = \tfrac34\Delta F$, so each $g_{nm}$ is localised within about one pixel in
frequency and a few $\Delta T$ in time.

### 3.3 Coefficients

The coefficients are the projections of the data on the basis,

```math
w_{c,nm} = \sum_{l=0}^{N-1} x_c(t_l)\, g_{nm}(t_l).
```

They are computed in the frequency domain (Cornish 2020 Eq. 13). For each row $m$, the FFT of $x_c$ is multiplied by
$\tilde\Phi$ centred on $f_m$, an $N_t$-point inverse FFT gives one value per time pixel, and the real or imaginary part
is kept according to the parity of $n + m$. Cost: $N_f$ FFTs of length $N_t$. Code:
[_forward_freq](../bahamas/wavelet/wdm.py#L102), wrappers [forward_time](../bahamas/wavelet/wdm.py#L194) and
[forward_freq](../bahamas/wavelet/wdm.py#L158); inverse [inverse_time](../bahamas/wavelet/wdm.py#L212).

The normalisation makes the basis **orthonormal** on the sampled series,
$\sum_l g_{nm}(t_l)\thinspace g_{n'm'}(t_l) = \delta_{nn'}\delta_{mm'}$, so $\sum_{nm} w^2_{c,nm} = \sum_l x_c^2(t_l)$ and the
transform is exactly invertible ([test_orthonormal_and_invertible](../tests/test_wdm.py#L11)). It agrees with
`pywavelet` coefficient by coefficient up to a factor $\sqrt2$ in their normalisation
([test_matches_pywavelet](../tests/test_wdm.py#L35)).

### 3.4 Variance of a coefficient

The likelihood needs the distribution of $w_{c,nm}$.

*Stationary white noise.* Samples of variance $\sigma_x^2$ have one-sided PSD $S = 2\Delta t\thinspace \sigma_x^2$. An
orthonormal transform maps independent $\mathcal N(0, \sigma_x^2)$ samples to independent $\mathcal N(0,\sigma_x^2)$
coefficients, so

```math
\mathrm{Var}(w_{nm}) = \sigma_x^2 = \frac{S}{2\Delta t}.
```

*Stationary coloured noise.* $w_{nm}$ sees the PSD through the squared spectral window of pixel $m$:

```math
\mathrm{Var}(w_{nm}) = \frac{\langle S\rangle_m}{2\Delta t},\qquad
\langle S\rangle_m = \sum_j W_j\, S(f_m + \delta_j\Delta F),\qquad
W_j \propto \big\lvert\tilde\Phi(2\pi\delta_j\Delta F)\big\rvert^2,\quad \sum_j W_j = 1,
```

with $\delta_j$ a grid of offsets in $\lvert\delta\rvert \lt \tfrac34$ (`band_samples` points per side,
[band_weights](../bahamas/wavelet/wdm.py#L227)). This matters for steep spectra: for an $f^{-4}$ spectrum the
band-averaged prediction passes a $\chi^2$ test and the pixel-centre value $S(f_m)$ fails
([test_band_average_matches_coloured_noise](../tests/test_wdm.py#L85)).

*Locally stationary (our data).* If $S_c(f, t)$ changes slowly over the support of $g_{nm}$, replace $S$ by
$S_c(f, t_n)$:

```math
\boxed{\;w_{c,nm} \sim \mathcal N\big(0,\ \sigma^2_{c,nm}\big),\qquad
\sigma^2_{c,nm} = \frac{P_c(t_n)\,\langle S_{\rm gal}\rangle_m + \langle S_{n,c}\rangle_m}{2\Delta t}\;}
```

and treat different pixels as independent (the diagonal approximation of Cornish 2020). Here $P_c$ varies over days and
a pixel is 10 h long. The approximation is checked on simulated data: at the injected parameters
$w^2/\sigma^2$ has mean 1 and variance 2, as $\chi^2_1$ requires
([test_whitened_coefficients](../tests/test_wavelet_model.py#L113)).

**Edges.** The FFT-based transform is periodic, so the first and last time pixels mix the end of the series with its
start. `edge_bins` pixels are dropped at each end (2 in the injection runs, after a check of the pixel variances in
`data/wavelet_injection/data_checks.py`).

---

## 4. The modulation model (the atoms)

**Code:** [WaveletModulation](../bahamas/wavelet/model.py#L268).

### 4.1 Definition

The log of the power modulation of channel $c$ is a constant plus a sum of $K_c$ atoms:

```math
\ln P_c(t) = b_c + \sum_{k=1}^{K_c} a_{c,k}\;\psi\Big(\frac{t - t_{c,k}}{\tau_{c,k}}\Big),
\qquad
\psi(x) = e^{-x^2/2}\ \ (\texttt{gaussian, default}),\qquad
\psi(x) = (1 - x^2)\,e^{-x^2/2}\ \ (\texttt{ricker}).
```

Each atom has three parameters, sampled as

| parameter | meaning | prior (injection runs) |
|---|---|---|
| $u_k = (t_k - t_0)/T$ | centre as a fraction of the span $T = N\Delta t$ | uniform on $[0, 1]$ |
| $\lambda_k = \log_{10}(\tau_k / 1\thinspace \mathrm{d})$ | width | uniform on $[\log_{10}5, \log_{10}60]$ (`width_range_days`) |
| $a_k$ | amplitude, in units of $\ln P$ | uniform on $[-3, 3]$ (`amp_range`) |

and the number of atoms $K_c$ has a uniform prior on $\lbrace 0, \dots, 8\rbrace$ (`min_wavelets`, `max_wavelets`), per
channel and independently. The atoms of a channel are exchangeable: the model does not depend on their order.

Writing the model in $\ln P$ keeps $P_c(t) \gt 0$ for any parameters, and makes an atom of amplitude $a$ a
multiplicative change $e^{a}$ of the galactic power at its centre. The time-domain shape of the atoms of the run-1
injection is shown under the injection in `data/wavelet_injection/outputs/run1/injection.png`.

### 4.2 The levels, and why only $10^{\rm amp}P_c(t)$ is measured

The galaxy enters the data only through the product $S_{\rm gal}(f)\thinspace P_c(t) \propto A_{\rm gal}\thinspace P_c(t)$, with
$A_{\rm gal} = 10^{\rm amp}$. Writing it out,

```math
A_{\rm gal}\,P_c(t) = 10^{\,{\rm amp}}\; e^{\,b_c}\; \exp\Big[\sum_k a_{c,k}\,\psi(\cdot)\Big]
= 10^{\,{\rm amp} + b_c/\ln 10}\; \exp\Big[\sum_k a_{c,k}\,\psi(\cdot)\Big].
```

* `amp` and $b_A$ appear only in the combination ${\rm amp} + b_A/\ln 10$, so $b_A$ is fixed to 0 and absorbed into
  `amp`. $b_E$ is then free and measures the E level relative to A (`level_E` $= b_E - b_A$). For an injection made
  with $b_A \ne 0$ the "effective" amplitude that the model can recover is
  $`{\rm amp}_{\rm eff} = {\rm amp} + b_A/\ln 10`$ (run 1: $-43.943 \to -43.171$).
* Even with $b_A = 0$ the split is not unique: an atom much wider than the 30 days is almost constant over the data,
  so it shifts $\ln P_c$ by a constant that `amp` can undo. $P_c(t)$ alone is not identified; the product is.

Plots of the recovered modulation therefore show the galactic power relative to the injected amplitude,

```math
10^{\,\Delta\log_{10}A}\,P_c(t),\qquad \Delta\log_{10}A = {\rm amp} - {\rm amp}_{\rm inj},
```

computed per posterior sample ([galactic_power](../bahamas/wavelet/sampler.py#L225)). This is the $\Delta\log_{10}A$
on the y-axis of `modulation_recovery.png`. At the injection it equals the injected $P_c(t)$.

### 4.3 The three modulation models

Selected with `inference.modulation`:

| `modulation` | $P_c(t)$ | parameters |
|---|---|---|
| `none` | $1$ (stationary galaxy, null model) | spectral only |
| `parametric` | sky envelope $M_c^2(t)$ of the chunked analysis, [parametric_modulation](../bahamas/wavelet/model.py#L252) | `lat`, `long`, `s1`, `s2`, `psi` (those with `bounds`) |
| `wavelets` | the atom model of Section 4.1 | $K_c$ and $(u_k, \lambda_k, a_k)$ per channel, `level_E` |

---

## 5. The wavelet-domain likelihood

**Code:** [WaveletLikelihood](../bahamas/wavelet/model.py#L147), [log_likelihood](../bahamas/wavelet/model.py#L227).

### 5.1 Per pixel

From the boxed result of Section 3.4, the coefficients of the analysed pixels are independent Gaussians:

```math
\ln\mathcal L = -\frac12 \sum_{c\in\lbrace A,E\rbrace}\ \sum_{n}\ \sum_{m}
\left[\frac{w^2_{c,nm}}{\sigma^2_{c,nm}} + \ln\big(2\pi\,\sigma^2_{c,nm}\big)\right],
```

the sums running over the analysed time pixels $n$ (edges dropped) and frequency pixels $m$ in $[f_1, f_2]$ (pixel
$m = 0$ excluded).

### 5.2 Coarse-grained: the Gamma likelihood

To reduce the number of terms, pixels are grouped:

* in frequency, the pixels in $[f_1, f_2]$ are split into $N_g$ contiguous log-spaced groups $\mathcal M_g$
  (config `freq_bins`, [frequency_groups](../bahamas/wavelet/model.py#L120));
* in time, the analysed pixels are split into $N_s$ blocks $\mathcal N_s$ of $n_\tau$ consecutive pixels
  (config `time_bin`).

A group $(s, g)$ holds $\nu_g = n_\tau\thinspace \lvert\mathcal M_g\rvert$ pixels. Within it the model variance is taken as
constant,

```math
\sigma^2_{c,sg} = \frac{P_c(\bar t_s)\,\bar S_{{\rm gal},g} + \bar S_{n,c,g}}{2\Delta t},
\qquad \bar t_s = \frac{1}{n_\tau}\sum_{n\in\mathcal N_s} t_n,
\qquad \bar S_g = \frac{1}{\lvert\mathcal M_g\rvert}\sum_{m\in\mathcal M_g}\langle S\rangle_m
```

([band_average](../bahamas/wavelet/model.py#L210)). This is justified when $P_c$ changes little over $n_\tau\Delta T$
(20 h here) and the PSD little across a group. The data enter only through the sum of squares

```math
Y_{c,sg} = \sum_{n\in\mathcal N_s}\ \sum_{m\in\mathcal M_g} w^2_{c,nm},
```

which is a sufficient statistic: $Y_{c,sg}/\sigma^2_{c,sg}$ is a sum of $\nu_g$ squared independent standard normals,
so $Y_{c,sg}/\sigma^2_{c,sg} \sim \chi^2_{\nu_g}$, i.e. $Y_{c,sg} \sim \mathrm{Gamma}\big(\text{shape } \nu_g/2,\
\text{scale } 2\sigma^2_{c,sg}\big)$. The likelihood is the product of these densities:

```math
\boxed{\;
\ln\mathcal L(\boldsymbol Y\mid\theta) = \sum_{c\in\lbrace A,E\rbrace}\ \sum_{s=1}^{N_s}\ \sum_{g=1}^{N_g}
\left[\Big(\frac{\nu_g}{2} - 1\Big)\ln Y_{c,sg} - \frac{\nu_g}{2}\ln\big(2\sigma^2_{c,sg}\big)
- \ln\Gamma\Big(\frac{\nu_g}{2}\Big) - \frac{Y_{c,sg}}{2\sigma^2_{c,sg}}\right]\;}
```

with $\theta$ all the model parameters: the spectral ones in $\sigma^2$ through $S_{\rm gal}$ and $S_{n,c}$, and the
modulation ones through $P_c$. This is exactly what the code evaluates: the data-only terms
$(\nu/2 - 1)\ln Y - (\nu/2)\ln 2 - \ln\Gamma(\nu/2)$ are precomputed once (`_const`), and per call it adds
$-\sum[(\nu_g/2)\ln\sigma^2 + Y/(2\sigma^2)]$.

With one pixel per group ($\nu_g = 1$, $Y = w^2$) the bracket is the Gaussian term of Section 5.1 minus
$\tfrac12\ln w^2$, a data-only constant (the Jacobian from $w$ to $w^2$). The two likelihoods then give identical
posteriors ([test_binned_likelihood_matches_pixel_likelihood_differences](../tests/test_wavelet_model.py#L124)).

---

## 6. Consistency with the Gamma likelihood of arXiv:2511.03604 (Eq. 9)

### 6.1 Their likelihood

Buscicchio, Pozzoli, Chirico & Sesana (2025) split the data into chunks, Fourier transform each chunk and average the
periodogram over log-spaced frequency bins. In their notation (we write their chunk index as $i$, because $c$ is our
channel index): $j \in \lbrace A, E\rbrace$ is the channel, $k$ a Fourier frequency, $m$ a coarse bin with $n_{b_m}$
Fourier frequencies. The Fourier amplitudes are normalised to the one-sided PSD (their Eq. 2, with $W = N$),

```math
\tilde d^{\,i}_{jk} = \sqrt{\frac{2}{N f_s}}\ \sum_{l} d^{\,i}_{jl}\,e^{-2\pi\imath kl/N},
\qquad \mathbb E\,\lvert\tilde d^{\,i}_{jk}\rvert^2 = S^{\,i}_{jk},
```

the averaged periodogram is (their Eq. 4)

```math
\hat P^{\,i}_{jm} = \frac{1}{n_{b_m}}\sum_{k\in\mathcal F_m}\lvert\tilde d^{\,i}_{jk}\rvert^2,
```

and the single-chunk likelihood is their **Eq. 9**:

```math
\ln\mathcal L_G^{\rm chunk}\big(\hat{\boldsymbol P}^{\,i}\mid\theta\big) =
-\sum_{j\in\lbrace A,E\rbrace}\ \sum_{m=1}^{n_g}\left[\ln\Gamma\Big(\frac{n_{b_m}}{2}\Big)
+ \frac{n_{b_m}}{2}\ln S^{\,i}_{jm}(\theta) - \Big(\frac{n_{b_m}}{2} - 1\Big)\ln\hat P^{\,i}_{jm}
+ \frac{n_{b_m}}{2}\,\frac{\hat P^{\,i}_{jm}}{S^{\,i}_{jm}(\theta)}\right],
```

summed over chunks $i$ (their Eq. 1), with the model PSD (their Eq. 10, without the extragalactic term)

```math
S^{\,i}_{jm}(\theta) = \overline M^{\,i}_j\, S^{\rm GF}_m + S^{(n)}_{jm}.
```

In words: $\hat P$ is the **data** (an averaged periodogram), $S$ is the **model** PSD (noise
$S^{(n)}$ plus the modulated galaxy $\overline M\thinspace S^{\rm GF}$), and $n_{b_m}$ is the number of averaged
values. Eq. 9 is the log-density of $\hat P$ for $\hat P = (S/n_{b_m})\thinspace \chi^2_{n_{b_m}}$: mean $S$, shape
$n_{b_m}/2$, rate $n_{b_m}/(2S)$ (the constant $\tfrac{n_{b_m}}{2}\ln\tfrac{n_{b_m}}{2}$, which does not depend on
$\theta$, is left out).

### 6.2 Our likelihood written in their variables

Define, for each of our groups, a **wavelet periodogram** and a **wavelet model PSD**, both in units of a one-sided
PSD:

```math
\hat P^{\rm wdm}_{c,sg} \equiv \frac{2\Delta t}{\nu_g}\,Y_{c,sg}
= \frac{1}{\nu_g}\sum_{n\in\mathcal N_s}\ \sum_{m\in\mathcal M_g} 2\Delta t\,w^2_{c,nm},
\qquad
S^{\rm wdm}_{c,sg} \equiv 2\Delta t\,\sigma^2_{c,sg} = P_c(\bar t_s)\,\bar S_{{\rm gal},g} + \bar S_{n,c,g}.
```

By Section 3.4, each $2\Delta t\thinspace w^2_{c,nm}$ has expectation $\langle S_c\rangle_m$, just as each
$\lvert\tilde d_{jk}\rvert^2$ has expectation $S_{jk}$. So $\hat P^{\rm wdm}$ is an average of PSD estimates, like
$\hat P$, and $\mathbb E\thinspace \hat P^{\rm wdm} = S^{\rm wdm}$.

Substitute $Y = \nu\hat P^{\rm wdm}/(2\Delta t)$ and $2\sigma^2 = S^{\rm wdm}/\Delta t$ in the bracket of the boxed
likelihood of Section 5.2 (indices $c, s, g$ dropped):

| term of our likelihood | becomes |
|---|---|
| $\big(\tfrac{\nu}{2} - 1\big)\ln Y$ | $\big(\tfrac{\nu}{2} - 1\big)\ln\hat P^{\rm wdm} + \big(\tfrac{\nu}{2} - 1\big)\ln\tfrac{\nu}{2\Delta t}$ |
| $-\tfrac{\nu}{2}\ln(2\sigma^2)$ | $-\tfrac{\nu}{2}\ln S^{\rm wdm} + \tfrac{\nu}{2}\ln\Delta t$ |
| $-\ln\Gamma\big(\tfrac{\nu}{2}\big)$ | $-\ln\Gamma\big(\tfrac{\nu}{2}\big)$ |
| $-\dfrac{Y}{2\sigma^2}$ | $-\dfrac{\nu}{2}\thinspace \dfrac{\hat P^{\rm wdm}}{S^{\rm wdm}}$ |

Adding the rows,

```math
\ln p(Y\mid\theta) = -\left[\ln\Gamma\Big(\frac{\nu_g}{2}\Big) + \frac{\nu_g}{2}\ln S^{\rm wdm}_{c,sg}
- \Big(\frac{\nu_g}{2} - 1\Big)\ln\hat P^{\rm wdm}_{c,sg} + \frac{\nu_g}{2}\,\frac{\hat P^{\rm wdm}_{c,sg}}{S^{\rm wdm}_{c,sg}}\right]
+ C_g,
\qquad C_g = \frac{\nu_g}{2}\ln\frac{\nu_g}{2} - \ln\frac{\nu_g}{2\Delta t}.
```

The bracket is **term by term the bracket of their Eq. 9**, under the dictionary

| arXiv:2511.03604 | this pipeline | role |
|---|---|---|
| chunk $i$ | time block $s$ | time index of a likelihood term |
| channel $j$ | channel $c$ | |
| coarse frequency bin $m$, set $\mathcal F_m$ | frequency group $g$, set $\mathcal M_g$ (and $\mathcal N_s$) | |
| $`\hat P^{\thinspace i}_{jm}`$, averaged periodogram | $`\hat P^{\rm wdm}_{c,sg} = 2\Delta t\thinspace Y_{c,sg}/\nu_g`$ | **data** |
| $`S^{\thinspace i}_{jm}(\theta) = \overline M^{\thinspace i}_j S^{\rm GF}_m + S^{(n)}_{jm}`$ | $`S^{\rm wdm}_{c,sg} = P_c(\bar t_s)\bar S_{{\rm gal},g} + \bar S_{n,c,g}`$ | **model**: noise + modulated galaxy |
| $`\overline M^{\thinspace i}_j`$, envelope averaged over the chunk | $P_c(\bar t_s)$, modulation at the block centre | galactic modulation |
| $`S^{\rm GF}_m`$ | $\bar S_{{\rm gal},g}$ | galactic spectrum |
| $`S^{(n)}_{jm}`$ | $\bar S_{n,c,g}$ | instrument noise |
| $n_{b_m}$, number of averaged values | $\nu_g$, number of averaged pixels | degrees of freedom (see 6.3) |

and $C_g$ depends only on $\nu_g$ and $\Delta t$, not on $\theta$ or the data. (It is the paper's omitted
$\tfrac{\nu}{2}\ln\tfrac{\nu}{2}$ plus the Jacobian $\ln\lvert dY/d\hat P^{\rm wdm}\rvert$ between the two data
variables.) Summing over $c, s, g$:

```math
\ln\mathcal L_{\rm ours}(\boldsymbol Y\mid\theta) = \sum_{s}\ \ln\mathcal L_G^{\rm Eq.\,9}
\big(\hat{\boldsymbol P}^{\rm wdm}_{s}\mid\theta\big)\Big\rvert_{n_{b_m}\to\nu_g,\ S\to S^{\rm wdm}}
+ \text{const}.
```

A constant offset does not change the posterior, so **our likelihood is the paper's Eq. 9 applied to wavelet
periodograms**, with time blocks in the role of chunks. Two modelling differences remain, both outside Eq. 9 itself:
the galaxy modulation is evaluated at the block centre instead of averaged over the chunk (a 20-h block against a
modulation that varies over days), and the PSD is averaged over the WDM spectral window (Section 3.4).

### 6.3 The degrees of freedom: where the two domains differ

The Gamma form with shape $n/2$ is exact when the averaged quantity is a mean of $n$ independent terms, each
$S\thinspace \chi^2_1$, i.e. $n$ **real** Gaussian degrees of freedom. The two domains differ here.

* **Wavelet domain.** Each $w_{c,nm}$ is one real Gaussian (Section 3.4), so $2\Delta t\thinspace w^2 = \langle S\rangle\thinspace \chi^2_1$
  and the shape $\nu_g/2$ with $\nu_g$ = number of pixels is exact.
* **Fourier domain.** Each $\tilde d_{jk}$ is a *complex* circular Gaussian: real and imaginary parts are independent
  with variance $S/2$ each, so $\lvert\tilde d_{jk}\rvert^2 = \tfrac S2\thinspace \chi^2_2$, two real degrees of freedom. A mean of
  $n_{b_m}$ of them is $\tfrac{S}{2n_{b_m}}\chi^2_{2n_{b_m}}$: shape $n_{b_m}$, not $n_{b_m}/2$. The exact chunked
  likelihood is Eq. 9 with $n_{b_m} \to 2n_{b_m}$. As printed (and as in BAHAMAS' `gamma_lik`), Eq. 9 assumes twice
  the actual variance of $\hat P$, $\mathrm{Var}\thinspace \hat P = 2S^2/n_{b_m}$ instead of $S^2/n_{b_m}$, so its posteriors are
  about $\sqrt2$ too wide. Run 2 measures $n_{b_m}\thinspace \mathrm{Var}(\hat P/S) = 1.04 \pm 0.04$ on the simulated data,
  confirming two degrees of freedom per frequency
  ([WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md)).

The degree-of-freedom count also agrees between the domains when it is written per unit of time-frequency area. A
chunk of duration $T_i$ has Fourier frequencies spaced $1/T_i$, so a bin of width $\delta f$ holds
$n_b = T_i\thinspace \delta f$ complex values, i.e. $2T_i\thinspace \delta f$ real degrees of freedom. The same rectangle holds
$(T_i/\Delta T)(\delta f/\Delta F) = 2T_i\thinspace \delta f$ WDM pixels (since $\Delta T\Delta F = \tfrac12$), each one real
degree of freedom. So the correct correspondence is

```math
\nu_g \;\longleftrightarrow\; 2\,n_{b_m},
```

the same information in both domains, as expected for two orthogonal transforms of the same data. The wavelet
likelihood uses the exact $\nu_g$; the chunked one, as printed, uses $n_{b_m}$ where $2n_{b_m}$ is exact.

---

## 7. How the fit is done

**Code:** [sampler.py](../bahamas/wavelet/sampler.py).

### 7.1 Posterior

For `modulation: 'wavelets'` the parameters are

* the spectral parameters $\theta_{\rm s}$: galaxy `alpha`, `amp`, `fknee`, `fr1`, `fr2`, noise `A`, `P`, and the
  level $b_E$ (those with `bounds` in the sources file; uniform priors within them);
* for each channel, the number of atoms $K_c$ and the atom parameters $\zeta_{c,k} = (u_{c,k}, \lambda_{c,k}, a_{c,k})$.

The posterior is

```math
p\big(\theta_{\rm s}, \lbrace K_c, \zeta_{c,1..K_c}\rbrace \mid \boldsymbol Y\big) \;\propto\;
\mathcal L\big(\boldsymbol Y\mid\theta_{\rm s}, \lbrace\zeta\rbrace\big)\ \pi(\theta_{\rm s})
\prod_{c\in\lbrace A,E\rbrace}\Big[\pi(K_c)\prod_{k=1}^{K_c}\pi(\zeta_{c,k})\Big],
```

with $\mathcal L$ the boxed likelihood of Section 5.2 and $P_c(t)$ from Section 4.1. Its dimension changes with
$K_c$, so it is sampled with reversible-jump MCMC.

### 7.2 One likelihood call

[WaveletPosterior.single](../bahamas/wavelet/sampler.py#L216) maps one point to $\ln\mathcal L$ in four steps:

1. $S_{\rm gal}$ and $S_{n,c}$ on the window sub-samples, band-averaged per group to $\bar S_{{\rm gal},g}$,
   $\bar S_{n,c,g}$ ([spectra](../bahamas/wavelet/sampler.py#L183));
2. $P_c(\bar t_s)$ from the atoms (inactive slots masked, see below);
3. $\sigma^2_{c,sg}$;
4. the boxed sum over $c, s, g$.

It is pure JAX: jit-compiled, `vmap`-ed over walkers (`WaveletPosterior.batch`) and differentiable
([test_jit_vmap_and_grad_agree](../tests/test_wavelet_model.py#L186)). Atoms are passed padded to `max_wavelets` with a
boolean mask, so array shapes are fixed. Inactive slots are replaced by a zero-amplitude atom before any arithmetic,
because eryn stores NaN there ([model.py:318](../bahamas/wavelet/model.py#L318)). An `optimization_barrier`
([sampler.py:222](../bahamas/wavelet/sampler.py#L222)) stops XLA from recomputing the spectra for every time block
(skipped on jax 0.4.30, where it is not public; results identical, just slower).

### 7.3 Reversible-jump MCMC (eryn)

[WaveletSampler](../bahamas/wavelet/sampler.py#L328). Branches: `psd` ($\theta_{\rm s}$, one leaf), `envelope` (sky
parameters, `parametric` only), and `mod_A`, `mod_E` (one leaf per atom, $K_c$ leaves). Moves:

* **within a model** (fixed $K$): stretch moves on `psd`; a Gaussian random walk on the atoms, with step
  `atom_step` (2%) of each prior width;
* **between models**: birth/death moves per channel (`DistributionGenerateRJ`). A birth draws a new atom from its
  prior $\pi(\zeta)$; a death removes a random atom. Because the new atom is drawn from its prior, its prior density
  cancels the proposal density and the acceptance probability reduces to the likelihood ratio times the ratio of
  move-selection probabilities. eryn includes the edge corrections at $K_{\min}$ and $K_{\max}$, so the prior on $K$
  is uniform ([test_prior_only_k_is_uniform](../tests/test_wavelet_rj.py#L32)).
* **parallel tempering**: `ntemps` copies of the ensemble at inverse temperatures $1 = \beta_0 \gt \beta_1 \gt \dots$
  sample $\mathcal L^{\beta}\pi$ and swap states; hot chains change $K$ more easily.

**Walkers.** The injection runs use **32 walkers per temperature × 4 temperatures = 128 walkers**. Only the 32
walkers of the cold chain ($\beta = 1$) sample the posterior; all results use them. Every walker starts at $K_c = 0$,
with $\theta_{\rm s}$ in a small ball around the injection; 2000 burn-in steps are discarded and 6000 stored.
Run 1 shows how the ensemble explores $K$ from the start, burn-in included (min–max band over the cold walkers):
`data/wavelet_injection/outputs/run1/K_trace_burnin.png`, described in
[WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md).

Two eryn details were needed for a correct reversible jump, both found by testing:

1. **$K = 0$ must be reachable.** eryn sets the log prior to $-\infty$ when a Gibbs-restricted RJ move empties every
   branch it acts on. Each RJ move is therefore grouped with the `psd` branch, whose leaf count is fixed
   ([sampler.py:398](../bahamas/wavelet/sampler.py#L398)). Without this, $K = 0$ was never visited.
2. **The prior on $K$ is uniform**, checked with a flat likelihood (test above).

### 7.4 Bayes factor versus the number of atoms

With a uniform prior on $K_c$, the Bayes factor between two values is their posterior odds, read from the chain:

```math
\ln\mathcal B(K, K') = \ln\frac{p(K_c = K\mid\boldsymbol Y)}{p(K_c = K'\mid\boldsymbol Y)},
\qquad p(K_c = K\mid\boldsymbol Y) \approx \frac{\#\lbrace\text{cold samples with } K_c = K\rbrace}{\#\lbrace\text{cold samples}\rbrace}.
```

$K'$ is the most probable value. Errors come from splitting the chain into consecutive blocks; never-visited values
get an upper limit. Code: [bayes_factors](../bahamas/wavelet/sampler.py#L279).

### 7.5 The reference fit (the run-1 injection)

Run 1 injects a modulation the atom model represents exactly: the least-squares atom fit to the true envelope
$M_c^2(t)$ (`data/wavelet_injection/fit_reference.py`). On an hourly grid $t_h$ over the 30 days it minimises, for
$K = 0, 1, \dots$,

```math
\frac{1}{N_h}\sum_h\Big[b_c + \sum_{k=1}^{K} a_k\,\psi\Big(\frac{t_h - t_k}{\tau_k}\Big) - \ln M_c^2(t_h)\Big]^2
+ \epsilon\sum_k a_k^2,\qquad \epsilon = 10^{-6},
```

with L-BFGS-B and random restarts, within the inference priors (centres kept 0.6 d from the ends). It keeps the
smallest $K$ with an rms error in $P$ below 1% (A: $K = 2$, E: $K = 1$). The small penalty picks the
smallest-amplitude solution along the level/width degeneracy. The fit is written in the inference parametrisation
$`{\rm amp}_{\rm eff} = {\rm amp} + b_A/\ln 10`$, $b_E \to b_E - b_A$ (Section 4.2).

### 7.6 NUTS for the fixed-dimension models

[run_nuts](../bahamas/wavelet/sampler.py#L495) samples `parametric` and `none` with NumPyro's NUTS, using gradients of
the same likelihood. NUTS cannot change $K$, so `wavelets` needs eryn.

---

## 8. Code overview

| Step | What happens | Code |
|---|---|---|
| Simulate | one continuous series per channel, $n_c + \sqrt{P_c}\thinspace g_c$ | [simulate.py](../bahamas/wavelet/simulate.py) |
| Transform | WDM to an $N_t \times N_f$ grid | [wdm.py](../bahamas/wavelet/wdm.py) |
| Likelihood | Gaussian per pixel or Gamma per group | [model.py](../bahamas/wavelet/model.py) |
| Sampling | eryn with reversible jump, or NUTS | [sampler.py](../bahamas/wavelet/sampler.py) |
| Pipeline | config-driven data generation, inference, plots | [pipeline.py](../bahamas/wavelet/pipeline.py), [plotting.py](../bahamas/wavelet/plotting.py) |
| CLI hooks | `bahamas_data` / `bahamas_inference` dispatch on `domain: 'wavelet'` | [run_data.py:28-33](../bahamas/utilities/run_data.py#L28-L33), [run_pe.py:35-46](../bahamas/utilities/run_pe.py#L35-L46) |
| Injection runs | 30-day injections, comparison with the chunked pipeline | [data/wavelet_injection/](../data/wavelet_injection/README.md) |

---

## 9. Validation

Tests: [tests/test_wdm.py](../tests/test_wdm.py), [tests/test_wavelet_model.py](../tests/test_wavelet_model.py),
[tests/test_wavelet_rj.py](../tests/test_wavelet_rj.py) (46 tests, about 2 minutes). They pass with the latest packages
(jax 0.6.2) and with the versions pinned in `requirements.txt` (jax 0.4.30, numpy 1.26, scipy 1.13).

**Transform**

| Check | Test |
|---|---|
| Orthonormal ($\sum w^2 = \sum x^2$) and exactly invertible (error $\sim 10^{-15}$), several grid shapes | [test_orthonormal_and_invertible](../tests/test_wdm.py#L11) |
| Agrees with `pywavelet` coefficient by coefficient (theirs is $\sqrt 2\times$ ours); `pywavelet` 0.2.6, 0.2.7, 0.2.10 | [test_matches_pywavelet](../tests/test_wdm.py#L35) |
| White noise gives unit-variance coefficients in every frequency row | [test_white_noise_variance](../tests/test_wdm.py#L57) |
| A sine-Gaussian lands in the right time-frequency pixel | [test_localisation](../tests/test_wdm.py#L68) |
| Band-averaged PSD predicts coloured-noise variances ($\chi^2$ test) | [test_band_average_matches_coloured_noise](../tests/test_wdm.py#L85) |

**Physics and likelihood**

| Check | Test |
|---|---|
| Time average of $P_c(t)$ equals the chunked-analysis envelope ($10^{-7}$) | [test_power_modulation_averages_to_chunk_envelope](../tests/test_wavelet_model.py#L52) |
| Simulated noise has the input PSD (Welch estimate) | [test_stationary_gaussian_matches_psd](../tests/test_wavelet_model.py#L61) |
| Modulated noise has pixel variances that follow $P(t)$ | [test_modulated_noise_variance_tracks_modulation](../tests/test_wavelet_model.py#L79) |
| At the injection, $w^2/\sigma^2$ has mean 1 and variance 2 ($\chi^2_1$) over a simulated year | [test_whitened_coefficients](../tests/test_wavelet_model.py#L113) |
| Gamma (grouped) likelihood equals the Gaussian one up to a constant | [test_binned_likelihood_matches_pixel_likelihood_differences](../tests/test_wavelet_model.py#L124) |
| The injected envelope beats a stationary galaxy and a wrong sky position by $\Delta\ln\mathcal L \gt 100$ | [test_likelihood_prefers_injected_modulation](../tests/test_wavelet_model.py#L172) |
| `jit`, `vmap` and `grad` agree with plain evaluation and finite differences | [test_jit_vmap_and_grad_agree](../tests/test_wavelet_model.py#L186) |
| eryn's batched interface gives the same values as single-walker evaluation | [test_eryn_vectorised_interface](../tests/test_wavelet_model.py#L219) |
| A short NUTS run brackets the injected values | [test_nuts_recovers_amplitude](../tests/test_wavelet_model.py#L249) |

**Independence of pixels (the diagonal approximation)**

The tests above check each pixel on its own (mean and variance of $w^2/\sigma^2$). The Gamma likelihood also needs the
$\nu_g$ pixels of a group to be **independent**: only then is $`Y_{c,sg} \sim \mathrm{Gamma}(\nu_g/2,\thickspace 2\sigma^2_{c,sg})`$,
which has

```math
\nu_g\thinspace\mathrm{Var}\left(\frac{Y_{c,sg}}{\mathbb E\thinspace Y_{c,sg}}\right) = 2 .
```

Correlated neighbours would give more than 2, i.e. fewer effective degrees of freedom than $\nu_g$, and posteriors that
are too narrow. [data/wavelet_injection/dof_check.py](../data/wavelet_injection/dof_check.py) measures this on 100
fresh realisations of the run-1 injection (30 d, $N_f$ = 3600, TDI gen 1, `edge_bins` = 2, `time_bin` = 2,
`freq_bins` = 200, i.e. 174 frequency groups with $\nu_g$ = 2 to 116). The injected modulation is made of wavelet
atoms, so $\mathbb E\thinspace Y = \nu_g\sigma^2$ at the injection is exact and only the correlations are tested. It also
measures the correlation of whitened pixels $w/\sigma$ with their neighbour in frequency $(n, m+1)$ and in time
$(n+1, m)$.

| band | groups | median $\nu_g$ | mean of $Y/\mathbb E\thinspace Y$ (expect 1) | $\nu_g\thinspace\mathrm{Var}(Y/\mathbb E\thinspace Y)$ (expect 2) | neighbour correlation, frequency / time |
|---|---|---|---|---|---|
| 0.1–0.3 mHz | 14 | 2 | 1.005 | 2.010 ± 0.024 | +0.003 / +0.001 |
| 0.3–1 mHz | 39 | 2 | 1.002 | 1.997 ± 0.010 | −0.002 / −0.003 |
| 1–3 mHz | 39 | 6 | 1.002 | 2.007 ± 0.009 | +0.000 / −0.000 |
| 3–10 mHz | 44 | 22 | 1.000 | 1.999 ± 0.006 | +0.000 / −0.000 |
| 10–29 mHz | 38 | 69 | 1.001 | 2.006 ± 0.007 | +0.000 / +0.000 |
| all | 174 | | 1.0015 | 2.003 | |

The effective number of degrees of freedom equals $\nu_g$ to 1% in every band, including the lowest frequencies, where
the spectrum is steepest and leakage between neighbours would show first. The largest single-pixel correlation is
0.03, about $3\sigma$ of the statistical error (0.009) over thousands of pixels, as expected from noise. So the shape
$\nu_g/2$ of the Gamma likelihood is right and the wavelet posteriors are not too narrow. Scope: data simulated from
the same model as the likelihood (no gaps or glitches), A and E independent by construction, and this configuration
only.

**Reversible jump**

| Check | Test |
|---|---|
| Flat likelihood: $p(K)$ uniform over $0..4$, $\ln\mathcal B \approx 0$ | [test_prior_only_k_is_uniform](../tests/test_wavelet_rj.py#L32) |
| Factorised likelihood with closed-form $p(K) \propto c^K$: reproduced to $\sim 1$% | [test_rj_matches_analytic_k_posterior](../tests/test_wavelet_rj.py#L51) |
| Data with known $K$ (0 and 2): fewer atoms excluded, more disfavoured | [test_posterior_prefers_true_number_of_wavelets](../tests/test_wavelet_rj.py#L115) |
| Started with no atoms, the sampler finds both injected features | [test_rj_detects_modulation_from_scratch](../tests/test_wavelet_rj.py#L132) |

**Injection runs.** 30-day injection and recovery, and the comparison with the chunked pipeline on the same data:
[WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md), [WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md).

---

## 10. How to run

```bash
uv sync --extra test          # or: pip install -e ".[test]"
pytest tests                  # about 2 minutes

cd template
bahamas_data      --config config_wavelet_quick.yaml --sources pe_galaxy_cyclo.yaml
bahamas_inference --config config_wavelet_quick.yaml --sources pe_galaxy_cyclo.yaml
```

Templates:

- [template/config_wavelet_quick.yaml](../template/config_wavelet_quick.yaml): laptop-sized, a few minutes. One year at
  $\Delta t = 1000$ s, analysed in $[10^{-4}, 4.9\times10^{-4}]$ Hz.
- [template/config_wavelet.yaml](../template/config_wavelet.yaml): realistic. One year at $\Delta t = 10$ s,
  $[10^{-4}, 0.029]$ Hz, $N_f = 4096$ ($\Delta F = 1.2\times10^{-5}$ Hz, $\Delta T = 11.4$ h), 200 log frequency
  groups, `time_bin: 4`. Every option is commented.

Outputs (in `folder_plot` and `inference.file_post`):

| File | Content |
|---|---|
| `result_*.npz` | chains (`psd_chain`, `envelope_chain`, `mod_<ch>_atoms/_inds/_nleaves`), `log_like`, Bayes factors, modulation quantiles |
| `wavelet_bayes_factor_*.png` | posterior on $K$ and $\ln\mathcal B$ versus $K$, per channel |
| `wavelet_modulation_*.png` | reconstructed $10^{\Delta\log_{10}A}P_c(t)$ with 50%/90% bands and the injection |
| `wavelet_k_trace_*.png` | mean $K$ versus stored step |
| `wavelet_corner_*.png` | corner plots of the spectral (and envelope) parameters |
| `wavelet_data_*.png` | wavelet power of the data |

Main options (`wavelet:` and `inference:` sections):

| Option | Meaning | Symbol |
|---|---|---|
| `domain: 'wavelet'` | selects this pipeline (default `'stft'`) | |
| `wavelet.Nf` | number of WDM frequency pixels (even) | $N_f$ |
| `wavelet.nx` | steepness of the Meyer window | $d$ |
| `wavelet.freq_bins` | number of log frequency groups (`null` = per pixel) | $N_g$ |
| `wavelet.time_bin` | time pixels per block (`1` = per pixel) | $n_\tau$ |
| `wavelet.edge_bins` | time pixels dropped at each end | |
| `inference.modulation` | `wavelets`, `parametric` or `none` | |
| `inference.sampler` | `eryn` (any model) or `NUTS` (`parametric`, `none`) | |
| `inference.min_wavelets`, `max_wavelets` | range of $K_c$ | $K_{\min}, K_{\max}$ |
| `inference.width_range_days`, `amp_range`, `level_range` | atom and level priors | $\tau_k$, $a_k$, $b_E$ |
| `inference.ntemps` | parallel-tempering temperatures | |

---

## 11. Performance

Measured on a laptop CPU (WSL):

- Realistic config, data generation: about 15 s and 1.1 GB peak memory ($3\times10^6$-sample series).
- Realistic config, likelihood with `time_bin: 4`: about 1.3–1.5 ms per walker in a batch of 64, against 4–11 ms for
  the first (NumPy) version. A full eryn run (32 walkers, 2 temperatures, 5000 steps) takes roughly 45 minutes.
- Injection runs (30 days, $N_f = 3600$, 32 walkers × 4 temperatures, 8000 steps): about 40 minutes on 8 cores.
- On CPU the gains of JAX come from compiling the whole likelihood, batching walkers, gradients (NUTS) and GPUs, not
  from the per-pixel arithmetic.

---

## 12. Limitations and open points

1. **RJ mixing from scratch.** New atoms are proposed from the prior, and there is no split/merge move. From $K = 0$,
   walkers can build one feature out of several partial atoms and stay at too-high $K$ for a while. Use tempering,
   long runs, and check the $K$ trace. Data-driven birth proposals (as in BayesWave) are the natural improvement.
2. **$K$ is only well defined where the galaxy dominates.** Where the galaxy is below the noise, extra atoms barely
   change the variance, so the posterior on $K$ flattens. Over 30 days of a smooth envelope, one to three atoms draw
   the same curve (run 1), so $K$ is not identifiable there either; $10^{\Delta\log_{10}A}P_c(t)$ is the quantity to
   compare.
3. **Atom centres are confined to the data span** ($u_k \in [0, 1]$). Run 1 suggests extending this, since the data
   would like an atom at or before the start.
4. **Diagonal likelihood.** WDM coefficients are treated as independent and the variance as constant within a group.
   This is standard for locally stationary noise. For the injection-run settings it is checked directly: neighbouring
   pixels are uncorrelated and the effective degrees of freedom of each group equal $\nu_g$ to 1%
   ([Section 9](#9-validation)). Other grids, or real data with gaps and glitches, would need the same check.
5. **Spectral shape in the quick config.** Its band lies below the galaxy's knee, so `alpha`, `fknee`, `fr1` and
   `fr2` are unconstrained there and NUTS takes very deep trees. Fix them (no `bounds`) for quick runs.
6. **The wavelet pipeline needs the JAX backend.** With `--use_jax False` it stops with a clear error.

---

## 13. Other changes outside `bahamas/wavelet/`

| File | Change |
|---|---|
| [bahamas/utilities/run_data.py](../bahamas/utilities/run_data.py#L28-L33) | dispatch to the wavelet data generation when `domain: 'wavelet'` |
| [bahamas/utilities/run_pe.py](../bahamas/utilities/run_pe.py#L35-L46) | same for inference; the wavelet package is only imported for wavelet configs |
| [bahamas/psd_response/modulation.py](../bahamas/psd_response/modulation.py#L8-L12) | `envelopes_gaussian` uses the backend `jnp` so it can be traced; results unchanged |
| [pyproject.toml](../pyproject.toml), [setup.py](../setup.py) | register `bahamas.wavelet`; optional `test` extra (`pytest`, `pywavelet`); `requires-python >= 3.10` (needed by `eryn` and jax) |
| [requirements.txt](../requirements.txt) | add `eryn` and `pywavelet==0.2.7` (0.2.8+ needs a newer astropy than the one pinned) |
| [README.md](../README.md) | short section pointing to this option |
| [template/config_wavelet.yaml](../template/config_wavelet.yaml), [template/config_wavelet_quick.yaml](../template/config_wavelet_quick.yaml) | new templates |

## References

- R. Buscicchio, F. Pozzoli, D. Chirico, A. Sesana, *The first year of LISA Galactic foreground*,
  [arXiv:2511.03604](https://arxiv.org/abs/2511.03604) (Eq. 9: the Gamma likelihood of Section 6)
- F. Pozzoli et al., *Bahamas: BAyesian inference with HAmiltonian Montecarlo for Astrophysical Stochastic
  background*, [arXiv:2506.22542](https://arxiv.org/abs/2506.22542)
- N. J. Cornish, *Time-frequency analysis of gravitational wave data*, Phys. Rev. D 102, 124038 (2020),
  [arXiv:2009.00043](https://arxiv.org/abs/2009.00043)
- M. C. Digman, N. J. Cornish, *LISA gravitational wave sources in a time-varying galactic stochastic background*,
  [arXiv:2212.04600](https://arxiv.org/abs/2212.04600)
- V. Necula, S. Klimenko, G. Mitselmakher, *Transient analysis with fast Wilson-Daubechies time-frequency transform*,
  J. Phys. Conf. Ser. 363, 012032 (2012)
- M. L. Katz et al., Eryn: [github.com/mikekatz04/Eryn](https://github.com/mikekatz04/Eryn)
- `pywavelet`: [pypi.org/project/pywavelet](https://pypi.org/project/pywavelet/)
