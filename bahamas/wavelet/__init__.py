"""
Wavelet-domain (WDM) analysis of cyclostationary LISA data, written in JAX:
- wdm (Wilson-Daubechies-Meyer transform)
- simulate (continuous time series with a modulated galactic foreground)
- model (wavelet-domain likelihood and modulation models)
- sampler (eryn reversible-jump sampling over the number of wavelet atoms)
- pipeline (config-driven data generation and inference)
"""
from bahamas.backend_context import get_backend_components, initialize_backend

# The spectral models in psd_strain/psd_response read the backend at import time.
# Default to JAX like psd_function does, before any of them is imported from here.
if get_backend_components()[0] is None:
    initialize_backend(use_jax=True)

import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)
