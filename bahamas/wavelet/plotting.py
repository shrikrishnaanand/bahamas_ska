"""
Plots for the wavelet-domain analysis.
"""
import numpy as np
import matplotlib.pyplot as plt

from bahamas.wavelet.model import DAY


def plot_spectrogram(data, f1, f2, path, nbins=150):
    """
    Wavelet power of each channel, averaged in log frequency bins.

    Args:
        data (WaveletData): Wavelet coefficients.
        f1, f2 (float): Frequency band in Hz.
        path (str): Output file.
        nbins (int): Number of log frequency bins for display.
    """
    from bahamas.wavelet.model import frequency_groups

    pixels, starts = frequency_groups(data.freqs, f1, f2, nbins)
    counts = np.diff(np.append(starts, len(pixels)))
    f_edges = np.append(data.freqs[pixels[starts]], data.freqs[pixels[-1]])
    t_edges = np.append(data.times, data.times[-1] + (data.times[1] - data.times[0])) / DAY

    fig, axes = plt.subplots(len(data.channels), 1, figsize=(10, 3.5 * len(data.channels)), sharex=True,
                             squeeze=False)
    for ax, w, name in zip(axes[:, 0], data.w, data.channel_names):
        power = np.add.reduceat(w[:, pixels] ** 2, starts, axis=1) / counts
        mesh = ax.pcolormesh(t_edges, f_edges, np.log10(power.T), shading='flat', rasterized=True)
        ax.set_yscale('log')
        ax.set_ylabel(f'Frequency [Hz] ({name})')
        fig.colorbar(mesh, ax=ax, label=r'$\log_{10} \langle w^2 \rangle$')
    axes[-1, 0].set_xlabel('Time [days]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bayes_factors(results, branches, path):
    """
    Posterior on the number of wavelet atoms and log Bayes factors versus ``K``.

    Args:
        results (dict): Sampler results.
        branches (list): RJ branch names.
        path (str): Output file.
    """
    fig, axes = plt.subplots(2, len(branches), figsize=(5 * len(branches), 7), squeeze=False, sharex='col')
    for j, b in enumerate(branches):
        bf = results[f'{b}_bayes']
        K = bf['K']
        axes[0, j].bar(K, bf['posterior'], color='steelblue', alpha=0.8)
        axes[0, j].set_ylabel('Posterior probability')
        axes[0, j].set_title(b)

        seen = np.isfinite(bf['lnB'])
        axes[1, j].errorbar(K[seen], bf['lnB'][seen], yerr=np.nan_to_num(bf['lnB_err'][seen]),
                            fmt='o-', color='k', capsize=3, label='visited')
        if np.any(~seen):
            axes[1, j].scatter(K[~seen], np.full((~seen).sum(), bf['lnB_upper']), marker='v', color='crimson',
                               label='upper limit (never visited)')
        axes[1, j].axhline(0, color='grey', lw=0.8)
        axes[1, j].set_xlabel('Number of wavelets K')
        axes[1, j].set_ylabel(rf'$\ln \mathcal{{B}}(K, K={bf["K_ref"]})$')
        axes[1, j].legend(fontsize=8)
        axes[1, j].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_modulation(times, draws, channel_names, path, truth=None):
    """
    Reconstructed galactic power modulation with 50% and 90% credible bands.

    Args:
        times (array): Times in seconds.
        draws (array): Posterior draws, shape (ndraws, nchannels, ntimes).
        channel_names (list): Channel labels.
        path (str): Output file.
        truth (array, optional): Injected modulation, shape (nchannels, ntimes).
    """
    t = times / DAY
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(10, 3.2 * len(channel_names)), sharex=True,
                             squeeze=False)
    for i, (ax, name) in enumerate(zip(axes[:, 0], channel_names)):
        q = np.percentile(draws[:, i], [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(t, q[0], q[4], color='steelblue', alpha=0.25, label='90%')
        ax.fill_between(t, q[1], q[3], color='steelblue', alpha=0.45, label='50%')
        ax.plot(t, q[2], color='navy', lw=1.2, label='median')
        if truth is not None:
            ax.plot(t, truth[i], color='crimson', lw=1.2, ls='--', label='injected')
        ax.set_ylabel(rf'$10^{{\Delta \rm amp}} P_{name}(t)$')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=4)
    axes[-1, 0].set_xlabel('Time [days]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_k_trace(results, branches, path):
    """Mean number of atoms across walkers versus step, for checking convergence."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    n = results['nsteps']
    for b in branches:
        ax.plot(results[f'{b}_nleaves'].reshape(n, -1).mean(axis=1), label=b)
    ax.set_xlabel('Stored step')
    ax.set_ylabel('Mean K over walkers')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_corner(chain, names, path, truths=None):
    """Corner plot of fixed-dimension parameters."""
    import corner

    fig = corner.corner(chain, labels=[n.split(':')[-1] for n in names], truths=truths, truth_color='crimson',
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, bins=30, smooth=0.9)
    fig.savefig(path, dpi=120)
    plt.close(fig)
