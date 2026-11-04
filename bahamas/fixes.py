"""
Switches for the bug fixes listed in reports/PAPER_BUGS.md.

Every fix is on by default. Setting one to false in the ``fixes`` section of the config restores the code as it was
before that fix, in case the fix turns out to be wrong:

    fixes:
      F1: true    # sinh instead of cosh in H_S;(3) of the Galactic envelope
      B2: false   # back to one degree of freedom per complex amplitude (Pozzoli et al. Eqs. 3 and 9 as printed)

``inference: legacy_dof: true`` is kept as an alias of ``fixes: B2: false``.

The switches are read when the model functions are traced, so ``configure`` is called once per run, before any
data are simulated or likelihoods evaluated (``SignalProcessor``, ``setting_inference.set`` and the wavelet
pipeline do this). Data and inference configs should carry the same ``fixes`` section. In scripts and tests,
use ``configure`` or the ``override`` context manager.
"""
import contextlib
import sys

from bahamas.logger_config import logger

FIXES = {
    'F1': 'sinh instead of cosh in the first H_S;(3) term of the Galactic envelope',
    'F2': 'exp(-2 sigma_+^2), exp(-6 sigma_+^2) damping on the 3b and 5b harmonics of the Galactic envelope',
    'B1': 'ERYN Gamma likelihood uses P_hat, not |P_hat|^2, and the factor n_b in the data term',
    'B2': 'two real degrees of freedom per complex amplitude in the Gamma and Whittle likelihoods',
    'B3b': 'A/E test-mass transfer function of Nam et al. (2023) Table II',
    'B4': 'linear binning of gapped data, and PSD normalisation with each segment\'s own length',
    'B5': 'gapped segments stay between gaps; excess gaps are dropped at random',
    'B6': 'alpha0/beta0 enter as a phase shift of A^2 - E^2, not as a rotation of the envelopes',
    'B7': 'phase of the n = 4 term of A^2 - E^2 in the point-source chunk average',
}

_active = {key: True for key in FIXES}


def enabled(key):
    """True if the fix ``key`` (e.g. 'F1', 'B2') is on."""
    return _active[key]


def configure(config=None):
    """
    Set the fixes from a config dict (its ``fixes`` section, and ``inference: legacy_dof``). Fixes that are not
    mentioned are on. Returns the active settings.
    """
    config = config or {}
    settings = {key: True for key in FIXES}
    unknown = set(config.get('fixes') or {}) - set(FIXES)
    if unknown:
        raise ValueError(f"Unknown fixes {sorted(unknown)}; known: {list(FIXES)}")
    settings.update({key: bool(value) for key, value in (config.get('fixes') or {}).items()})
    if (config.get('inference') or {}).get('legacy_dof', False):
        settings['B2'] = False
    _set(settings)
    return dict(_active)


@contextlib.contextmanager
def override(**settings):
    """Temporarily switch fixes on or off, e.g. ``with override(F1=False): ...``."""
    unknown = set(settings) - set(FIXES)
    if unknown:
        raise ValueError(f"Unknown fixes {sorted(unknown)}; known: {list(FIXES)}")
    previous = dict(_active)
    _set({**previous, **settings})
    try:
        yield
    finally:
        _set(previous)


def _set(settings):
    changed = settings != _active
    _active.update(settings)
    if changed:
        # Compiled functions keep the switches they were traced with
        if 'jax' in sys.modules:
            sys.modules['jax'].clear_caches()
        for key in FIXES:
            if not _active[key]:
                logger.warning(f"Fix {key} is OFF (pre-fix behaviour): {FIXES[key]}")
