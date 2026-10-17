"""
Term-by-term check of ``envelopes_gaussian`` (bahamas/psd_response/modulation.py) against Buscicchio et al. 2024,
arXiv:2410.08263, Appendix A (equation lines quoted from the arXiv TeX source, main.tex).

Three layers:

1. The paper's point-source coefficients h_{S;(n)}(b), h_{D;(n)}(b) (App. A.1) against the paper's own detector
   model: arm vectors L_i(t) from the triad x, y, z(t), h_X = (L3 L3 - L2 L2) : h, A = (Z - X)/sqrt2,
   E = (X - 2Y + Z)/sqrt6, inclination average 56/15 C^2 + 8/3 S^2, numerical polarisation average.
2. The paper's printed Gaussian-averaged groups H_{S;(n)}, H_{D;(n)} (App. A.2) against the Gaussian average of
   10240 h(b) cos(b) computed independently: Fourier series of h(b) cos(b) in b, then the paper's characteristic
   function  E[e^{i k b} e^{i m l}] = exp[-(m^2 + k^2) sS/4 + (m^2 - k^2) sD cos2d / 4 + m k sD sin2d / 2] e^{ikb_M}
   e^{iml_M} (with the sign of the last term as used by the printed groups, see CROSS_SIGN).
3. The bahamas code, statement by statement (each ``Sum += ...`` / ``Diff += ...`` is one group), against both:
   before the fix (commit 4db1cc9) and as it is now.

Every comparison is made per group and per b-harmonic (the cos b / cos 3b / cos 5b terms), so a disagreement points
at one printed term or one line of code.

Paper transcription: ``PAPER`` below lists every term as printed (coefficient, b-harmonic k, hyperbolic argument in
units of sigma_s^2, sign of the sinh part); the damping e^{-2 sigma_+^2} (k = 3) and e^{-6 sigma_+^2} (k = 5) is on
every printed term. Two printed typos are flagged in ``PRINTED_AS``.

    python paper_terms_check.py
"""
import inspect

import numpy as np

import sky  # noqa: F401  (sets the JAX backend before bahamas.psd_response is imported)
import bahamas.psd_response.modulation as modulation
import envelope_check

SQ3 = np.sqrt(3.0)

# Two conventions of the paper that are not consistent with each other as printed (found by this script):
# - CROSS_SIGN: the printed cosh/sinh terms follow exp[- m k sD sin2d / 2] in the characteristic function, the
#   opposite sign of the printed one (equivalent to delta -> -delta). The derivation uses the printed groups' sign.
# - LBAR_OFFSET: with the arm vectors L_1, L_2, L_3 as printed (alpha0 = beta0 = 0), the detector gives
#   <h_A>^2 - <h_E>^2 = sum h_D;(n) cos(n DeltaL + 4 (l + pi/4)), not 4 (l + pi/12) as printed: a 30 deg
#   arm-labelling (beta0) convention. The groups H_D;(n) and the code use pi/12.
CROSS_SIGN = -1
LBAR_OFFSET_ARMS = np.pi / 4

# ---------------------------------------------------------------------------------------------------------------
# Paper, App. A.1: point-source coefficients (main.tex lines 784-805)
# ---------------------------------------------------------------------------------------------------------------
h_S = {
    0: lambda c, s: 9 / 320 * (328 + 152 * c ** 2 - 37 * c ** 4),
    1: lambda c, s: -9 * SQ3 / 40 * c * s * (52 + 5 * c ** 2),
    2: lambda c, s: 81 / 80 * c ** 2 * (10 - c ** 2),
    3: lambda c, s: -27 * SQ3 / 40 * s * c ** 3,
    4: lambda c, s: 81 / 320 * c ** 4,
}
h_D = {
    4: lambda c, s: -81 / 320 * (8 - 40 * c ** 2 + 35 * c ** 4),
    3: lambda c, s: -81 * SQ3 / 80 * s * c * (4 - 7 * c ** 2),
    5: lambda c, s: 27 * SQ3 / 80 * s * c * (4 - 7 * c ** 2),
    2: lambda c, s: -243 / 160 * c ** 2 * (6 - 7 * c ** 2),
    6: lambda c, s: -27 / 160 * c ** 2 * (6 - 7 * c ** 2),
    1: lambda c, s: -243 * SQ3 / 80 * s * c ** 3,
    7: lambda c, s: 9 * SQ3 / 80 * s * c ** 3,
    0: lambda c, s: -729 / 640 * c ** 4,
    8: lambda c, s: -9 / 640 * c ** 4,
}

# ---------------------------------------------------------------------------------------------------------------
# Paper, App. A.2: printed groups (main.tex lines 852-929).
# Odd n ("sin type"):  coef e^{..} [cosh(h ss) sin kb cos(arg) + sgn sinh(h ss) cos kb sin(arg)]
# Even n ("cos type"): coef e^{..} [cosh(h ss) cos kb cos(arg) + sgn sinh(h ss) sin kb sin(arg)]
# arg = n DeltaPhi for S, n DeltaPhi + 4 lbar for D.  Prefactor: (sqrt3 for odd n) exp(a sS + c sc).
# ---------------------------------------------------------------------------------------------------------------
PAPER = {
    # group: (sqrt3?, a, c, [(coef, k, h, sgn), ...])
    ('S', 0): (False, -1 / 4, -1 / 4, [(120636, 1, 0, 0), (7614, 3, 0, 0), (-666, 5, 0, 0)]),
    ('S', 1): (True, -1 / 2, 0, [(-31392, 1, 1 / 2, +1), (-32112, 3, 3 / 2, +1), (-720, 5, 5 / 2, +1)]),
    ('S', 2): (False, -5 / 4, 3 / 4, [(71280, 1, 1, -1), (22680, 3, 3, -1), (-648, 5, 5, -1)]),
    ('S', 3): (True, -5 / 2, 2, [(-864, 1, 3 / 2, +1), (-1296, 3, 9 / 2, +1), (-432, 5, 15 / 2, +1)]),
    ('S', 4): (False, -17 / 4, 15 / 4, [(1620, 1, 2, -1), (810, 3, 6, -1), (162, 5, 10, -1)]),
    ('D', 4): (False, -1 / 4, -1 / 4, [(324, 1, 0, 0), (-240, 3, 0, 0), (-5670, 5, 0, 0)]),
    ('D', 3): (True, -1 / 2, 0, [(-1296, 1, 1 / 2, -1), (3240, 3, 3 / 2, -1), (4536, 5, 5 / 2, -1)]),
    ('D', 5): (True, -1 / 2, 0, [(432, 1, 1 / 2, +1), (-1080, 3, 3 / 2, +1), (-1512, 5, 5 / 2, +1)]),
    ('D', 2): (False, -5 / 4, 3 / 4, [(-1944, 1, 1, +1), (10692, 3, 3, +1), (6804, 5, 5, +1)]),
    ('D', 6): (False, -5 / 4, 3 / 4, [(-216, 1, 1, -1), (1188, 3, 3, -1), (756, 5, 5, -1)]),
    ('D', 1): (True, -5 / 2, 2, [(-3888, 1, 3 / 2, -1), (-5832, 3, 9 / 2, -1), (-1944, 5, 15 / 2, -1)]),
    ('D', 7): (True, -5 / 2, 2, [(144, 1, 3 / 2, +1), (216, 3, 9 / 2, +1), (72, 5, 15 / 2, +1)]),
    ('D', 0): (False, -17 / 4, 15 / 4, [(-7290, 1, 2, +1), (-3645, 3, 6, +1), (-729, 5, 10, +1)]),
    ('D', 8): (False, -17 / 4, 15 / 4, [(-90, 1, 2, -1), (-45, 3, 6, -1), (-9, 5, 10, -1)]),
}
# Printed forms that are read with an interpretation here
PRINTED_AS = {
    ('S', 3): 'k = 3, 5 sinh parts print "sin Delta 3 phi_L"; read as sin 3 DeltaPhi_L',
    ('D', 4): 'printed phase cos(4 phi_L + 4 lbar_M), with phi_L (no Delta) not defined; '
              'evaluated both as printed (phi_L = Omega t) and as 4 DeltaPhi_L',
}
# Order of the Diff statements in the code
CODE_D_ORDER = [4, 3, 5, 2, 6, 1, 7, 0, 8]


def paper_group(key, k_only, p, d4_literal=False):
    """Printed group ``key`` restricted to b-harmonic ``k_only``."""
    sq3, a, c, terms = PAPER[key]
    kind, n = key
    pre = (SQ3 if sq3 else 1.0) * np.exp(a * p['sS'] + c * p['sc'])
    arg = n * p['Dphi'] + (4 * p['lbar'] if kind == 'D' else 0.0)
    if key == ('D', 4) and d4_literal:
        arg = 4 * p['Omega_t'] + 4 * p['lbar']
    damp = {1: 1.0, 3: np.exp(-2 * p['sp']), 5: np.exp(-6 * p['sp'])}
    total = 0.0
    for coef, k, h, sgn in terms:
        if k != k_only:
            continue
        kb = k * p['b']
        ch, sh = np.cosh(h * p['ss']), np.sinh(h * p['ss'])
        if n % 2:
            val = ch * np.sin(kb) * np.cos(arg) + sgn * sh * np.cos(kb) * np.sin(arg)
        else:
            val = ch * np.cos(kb) * np.cos(arg) + sgn * sh * np.sin(kb) * np.sin(arg)
        if kind == 'S' and n == 0 or kind == 'D' and n == 4:
            val = np.cos(kb) * np.cos(arg)          # no l-dependence: no hyperbolic factor
        total += coef * damp[k] * val
    return pre * total


# ---------------------------------------------------------------------------------------------------------------
# Layer 2: independent Gaussian average of 10240 h(b) cos b
# ---------------------------------------------------------------------------------------------------------------
def fourier_b(f, kmax=8, N=64):
    """Complex Fourier coefficients g_k of f(b) (a trigonometric polynomial), k = -kmax..kmax."""
    b = 2 * np.pi * np.arange(N) / N
    g = np.fft.fft(f(np.cos(b), np.sin(b)) * np.cos(b)) / N
    return {k: g[k % N] for k in range(-kmax, kmax + 1)}


def char(k, m, p):
    """Paper's characteristic function of the (b, l) Gaussian, without the phase e^{i k b_M + i m l_M}."""
    return np.exp(-(m ** 2 + k ** 2) * p['sS'] / 4 + (m ** 2 - k ** 2) * p['sD'] * p['cos2d'] / 4
                  + CROSS_SIGN * m * k * p['sD'] * p['sin2d'] / 2)


def derived_group(key, k_only, p):
    """10240 x Gaussian average of h_{key}(b) cos b cos(phase), restricted to b-harmonics +-k_only."""
    kind, n = key
    h = (h_S if kind == 'S' else h_D)[n]
    g = fourier_b(h)
    # cos(n DeltaL + [4 lbar_S]) with DeltaL = Omega t - l_S:  l-harmonic m, constant phase c0
    if kind == 'S':
        m, c0 = -n, n * p['Omega_t']
    else:
        m, c0 = 4 - n, n * p['Omega_t'] + np.pi / 3
    total = 0.0
    for k in (k_only, -k_only):
        # cos(X) = (e^{iX} + e^{-iX}) / 2 with X = m l + c0; average of e^{i(k b)} e^{+-i(m l + c0)}
        for sgn in (+1, -1):
            total += 0.5 * g[k] * char(k, sgn * m, p) * np.exp(1j * (k * p['b'] + sgn * (m * p['l'] + c0)))
    return 10240 * total.real


# ---------------------------------------------------------------------------------------------------------------
# Layer 1: the h coefficients against the paper's detector model
# ---------------------------------------------------------------------------------------------------------------
def detector_hS_hD(b, l, t, Omega=1.0, npsi=64):
    """Inclination- and polarisation-averaged <h_A>^2 +- <h_E>^2 from the paper's arm vectors (alpha0 = beta0 = 0)."""
    w = Omega * t
    x = np.array([(3 - np.cos(2 * w)) / 4, -np.sin(2 * w) / 4, SQ3 / 2 * np.cos(w)])
    y = np.array([-np.sin(2 * w) / 4, (3 + np.cos(2 * w)) / 4, SQ3 / 2 * np.sin(w)])
    L1, L2, L3 = -x / 2 - SQ3 / 2 * y, x, -x / 2 + SQ3 / 2 * y
    DX = np.outer(L3, L3) - np.outer(L2, L2)
    DY = np.outer(L1, L1) - np.outer(L3, L3)
    DZ = np.outer(L2, L2) - np.outer(L1, L1)
    DA, DE = (DZ - DX) / np.sqrt(2), (DX - 2 * DY + DZ) / np.sqrt(6)
    pv = np.array([np.sin(b) * np.cos(l), np.sin(b) * np.sin(l), -np.cos(b)])
    qv = np.array([np.sin(l), -np.cos(l), 0.0])
    ep, ex = np.outer(pv, pv) - np.outer(qv, qv), np.outer(pv, qv) + np.outer(qv, pv)
    out = []
    for D in (DA, DE):
        acc = 0.0
        for psi in np.pi * np.arange(npsi) / npsi:
            e_plus = ep * np.cos(2 * psi) - ex * np.sin(2 * psi)
            e_cross = ep * np.sin(2 * psi) + ex * np.cos(2 * psi)
            # h = -2 (1 + cos^2 i) e+ cos 2phi + 4 cos i ex sin 2phi  ->  C = -D:e+, S = D:ex
            C, S = -np.sum(D * e_plus), np.sum(D * e_cross)
            acc += 56 / 15 * C ** 2 + 8 / 3 * S ** 2
        out.append(acc / npsi)
    return out[0] + out[1], out[0] - out[1]


def layer1(rng, ntests=200):
    worst_S = worst_D = 0.0
    for _ in range(ntests):
        b, l, t = rng.uniform(-1.5, 1.5), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        hS, hD = detector_hS_hD(b, l, t)
        dl = t - l
        sS = sum(f(np.cos(b), np.sin(b)) * np.cos(n * dl) for n, f in h_S.items())
        sD = sum(f(np.cos(b), np.sin(b)) * np.cos(n * dl + 4 * (l + LBAR_OFFSET_ARMS)) for n, f in h_D.items())
        worst_S, worst_D = max(worst_S, abs(hS - sS)), max(worst_D, abs(hD - sD))
    return worst_S, worst_D


# ---------------------------------------------------------------------------------------------------------------
# Layer 3: the code, statement by statement
# ---------------------------------------------------------------------------------------------------------------
def code_statements(src=None):
    """Preamble source and the list of (name, expression) of the Sum / Diff statements, in code order."""
    src = inspect.getsource(modulation.envelopes_gaussian) if src is None else src
    body = src[src.index('"""', src.index('"""') + 3) + 3:]
    pre, rest = body.split('overall_factor = 1. / 10240.')
    stmts, cur = [], None
    for line in rest.split('\n'):
        s = line.strip()
        if s.startswith(('Sum =', 'Sum +=', 'Diff =', 'Diff +=')):
            name, expr = s.split('=', 1)
            cur = [name.replace('+', '').strip(), expr.strip()]
            stmts.append(cur)
        elif cur is not None and s and not s.startswith(('Sum *=', 'Diff *=', '#', 'A0', 'E0', 'sDect', 'cDect',
                                                          'A =', 'E =', 'return')):
            cur[1] += ' ' + s
        if s.startswith(('Sum *=', 'Diff *=')):
            cur = None
    for st in stmts:                                     # drop a trailing unmatched ')' of the last line, if any
        while st[1].count('(') < st[1].count(')'):
            st[1] = st[1][:st[1].rindex(')')]
    sums = [e for n, e in stmts if n == 'Sum']
    diffs = [e for n, e in stmts if n == 'Diff']
    keys = [('S', n) for n in range(5)] + [('D', n) for n in CODE_D_ORDER]
    return pre, dict(zip(keys, sums + diffs))


def code_group(pre, expr, k_only, p):
    ns = {'jnp': np}
    args = dict(EclipticLatitude=np.sin(p['b']), EclipticLongitude=p['l'], Sigma1=p['s1'], Sigma2=p['s2'],
                sinPsi=p['sin2d'], LISA_Orbital_Freq=1 / (2 * np.pi), t=p['Omega_t'], alpha0=0., beta0=0.)
    ns.update(args)
    exec(pre.replace('\n    ', '\n'), ns)
    for k in (1, 3, 5):                                  # keep one b-harmonic only
        if k != k_only:
            for nm in ('s{}bM', 'c{}bM'):
                ns[nm.format('' if k == 1 else k)] = 0.0
    return eval(expr, ns)


def params(rng, small=False):
    s1, s2 = rng.uniform(0.02, 0.3, 2) * (0.1 if small else 1)
    two_d = rng.uniform(-1.4, 1.4)                       # cos 2 delta > 0 (the code's sqrt branch)
    b, l, wt = rng.uniform(-1.2, 1.2), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    sS, sD = s1 + s2, s1 - s2
    return dict(s1=s1, s2=s2, sS=sS, sD=sD, cos2d=np.cos(two_d), sin2d=np.sin(two_d), sc=sD * np.cos(two_d),
                ss=sD * np.sin(two_d), sp=sS + sD * np.cos(two_d), b=b, l=l, lbar=l + np.pi / 12,
                Omega_t=wt, Dphi=wt - l)


def main():
    rng = np.random.default_rng(2024)
    wS, wD = layer1(rng)
    print(f'Layer 1  point-source h coefficients vs the paper detector model: max |diff| S {wS:.1e}, D {wD:.1e}\n')

    pre, code = code_statements(envelope_check.code_version(source=True))      # before the fix
    pre_f, code_f = code_statements()                                          # current code
    print('Layers 2-3, per group and b-harmonic: max over 200 random parameter sets of |X - derived| / scale')
    print('  derived = independent Gaussian average of the paper\'s h; scale = max |derived| over the sets\n')
    print(f"{'group':7s} {'k':>2s} {'paper printed':>14s} {'code before fix':>16s} {'current code':>13s}")
    sets = [params(rng) for _ in range(200)]
    rows = []
    for key in [('S', n) for n in range(5)] + [('D', n) for n in CODE_D_ORDER]:
        for k in (1, 3, 5):
            der = np.array([derived_group(key, k, p) for p in sets])
            pap = np.array([paper_group(key, k, p) for p in sets])
            cod = np.array([code_group(pre, code[key], k, p) for p in sets])
            fix = np.array([code_group(pre_f, code_f[key], k, p) for p in sets])
            scale = np.abs(der).max()
            e_pap, e_cod, e_fix = (np.abs(x - der).max() / scale for x in (pap, cod, fix))
            flag = '' if max(e_pap, e_fix) < 1e-9 else '  <--'
            rows.append((key, k, e_pap, e_cod, e_fix))
            print(f'H_{key[0]};({key[1]}) {k:2d} {e_pap:14.1e} {e_cod:16.1e} {e_fix:13.1e}{flag}')
    lit = np.array([paper_group(('D', 4), k, p, d4_literal=True) for p in sets for k in (1,)])
    der = np.array([derived_group(('D', 4), 1, p) for p in sets])
    print(f"\nH_D;(4) with the printed phase cos(4 phi_L + 4 lbar), phi_L = Omega t: "
          f"|diff| / scale = {np.abs(lit - der).max() / np.abs(der).max():.2f}")
    for key, note in PRINTED_AS.items():
        print(f'note {key}: {note}')


if __name__ == '__main__':
    main()
