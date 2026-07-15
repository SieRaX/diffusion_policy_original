"""Figures for the dense-eval analysis. Pure matplotlib (Agg); consumes arrays
already loaded from the npz. Every function saves a file and returns its path."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _model_curve(c, b, lam, C):
    x = np.arange(C, dtype=float)
    return x, c + b * np.exp(-lam * x)


def plot_lambda_c_distributions(lam, c, success, out_dir):
    """Histograms of λ and c, plus a λ-vs-c scatter (log axes)."""
    path = os.path.join(out_dir, 'lambda_c_distributions.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(lam, bins=50, color='C0')
    axes[0].set_title('fitted λ (rate)')
    axes[0].set_xlabel('λ'); axes[0].set_ylabel('# states')
    axes[1].hist(c, bins=50, color='C1')
    axes[1].set_title('fitted c (floor)')
    axes[1].set_xlabel('c'); axes[1].set_ylabel('# states')
    # scatter, log axes (guard non-positive)
    lam_p = np.clip(lam, 1e-6, None)
    c_p = np.clip(c, 1e-9, None)
    axes[2].scatter(lam_p[success], c_p[success], s=6, alpha=0.4, label='fit ok')
    if np.any(~success):
        axes[2].scatter(lam_p[~success], c_p[~success], s=6, alpha=0.4,
            color='red', label='fallback')
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_xlabel('λ'); axes[2].set_ylabel('c')
    axes[2].set_title('λ vs c'); axes[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_example_curves(scalar_CS, epochs, c, b, lam, out_dir):
    """20 example states spanning λ quantiles (5 per quartile), fitted overlay."""
    path = os.path.join(out_dir, 'example_curves.png')
    C, S = scalar_CS.shape
    order = np.argsort(lam)
    chosen = []
    for q in range(4):
        lo = int(q * S / 4)
        hi = int((q + 1) * S / 4)
        grp = order[lo:hi]
        if len(grp) == 0:
            continue
        pick = np.linspace(0, len(grp) - 1, num=min(5, len(grp))).round().astype(int)
        chosen.extend(grp[pick].tolist())

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(C, dtype=float)
    for s in chosen:
        line, = ax.plot(x, np.clip(scalar_CS[:, s], 1e-12, None),
            marker='.', linewidth=0.8, alpha=0.7)
        _, yfit = _model_curve(c[s], b[s], lam[s], C)
        ax.plot(x, np.clip(yfit, 1e-12, None), color=line.get_color(),
            linestyle='--', linewidth=1.0, alpha=0.9)
    ax.set_yscale('log')
    ax.set_xlabel('checkpoint index i'); ax.set_ylabel('MSE (log)')
    ax.set_title('Example per-state MSE vs checkpoint (dashed = fit), λ-spanning')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_episode_timeline(timesteps, scalar_final, first_final, lam, designated_ts, out_dir):
    """Final-checkpoint MSE (full-chunk & first-executed) and fitted λ vs env
    timestep for the designated episode; quarter points marked."""
    path = os.path.join(out_dir, 'episode_timeline.png')
    order = np.argsort(timesteps)
    t = np.asarray(timesteps)[order]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, scalar_final[order], marker='.', label='MSE (full chunk)')
    axes[0].plot(t, first_final[order], marker='.', label='MSE (first executed)')
    axes[0].set_yscale('log'); axes[0].set_ylabel('final-ckpt MSE (log)')
    axes[0].legend(); axes[0].set_title('Designated-episode timeline')
    axes[1].plot(t, lam[order], marker='.', color='C2')
    axes[1].set_ylabel('fitted λ'); axes[1].set_xlabel('env timestep t')
    for dt in designated_ts:
        for ax in axes:
            ax.axvline(dt, color='k', linestyle=':', alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_cross_variant_timeline(a, b, out_dir):
    """Overlay two variants' designated-episode timelines (same task+abs, low_dim
    vs image). ``a`` and ``b`` are dicts with keys t, scalar_final, lam, label."""
    path = os.path.join(out_dir, 'cross_variant_timeline.png')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for d in (a, b):
        o = np.argsort(d['t'])
        axes[0].plot(np.asarray(d['t'])[o], d['scalar_final'][o], marker='.', label=d['label'])
        axes[1].plot(np.asarray(d['t'])[o], d['lam'][o], marker='.', label=d['label'])
    axes[0].set_yscale('log'); axes[0].set_ylabel('final-ckpt MSE (log)')
    axes[0].legend(); axes[0].set_title('Cross-variant designated-episode timeline')
    axes[1].set_ylabel('fitted λ'); axes[1].set_xlabel('env timestep t'); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path
