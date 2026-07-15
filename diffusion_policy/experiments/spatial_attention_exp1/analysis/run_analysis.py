"""Analysis entry point (ONE entry point). Consumes ONLY a dense-eval npz (+ its
metadata) and writes figures + a markdown summary. Re-runnable standalone on any
past run's artifacts.

Usage (from repo root):
    python -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
      npz=outputs/exp1_lift_lowdim_abs/dense_eval.npz \
      output_dir=outputs/exp1_lift_lowdim_abs
    # cross-variant overlay (same task+abs, other obs variant):
    #   ... npz=<image.npz> npz_compare=<lowdim.npz> output_dir=<dir>
"""
import os
import pathlib

import hydra
import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.experiments.spatial_attention_exp1.analysis import fit as fitmod
from diffusion_policy.experiments.spatial_attention_exp1.analysis import plots


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _designated_episode_timeline(data, c, lam):
    """Return (t, scalar_final, first_final, lam) for the designated-episode states."""
    ep = int(data['designated_episode_index'])
    mask = data['state_episode'] == ep
    t = data['state_timestep'][mask]
    scalar_final = data['scalar'][-1][mask]
    first_final = data['first'][-1][mask]
    lam_ep = lam[mask]
    return t, scalar_final, first_final, lam_ep


def _q(a):
    a = np.asarray(a, dtype=float)
    return {
        'mean': float(np.mean(a)), 'median': float(np.median(a)),
        'p10': float(np.quantile(a, 0.1)), 'p90': float(np.quantile(a, 0.9)),
        'min': float(np.min(a)), 'max': float(np.max(a)),
    }


def analyze(npz_path, output_dir, npz_compare=None):
    os.makedirs(output_dir, exist_ok=True)
    data = _load(npz_path)
    scalar = data['scalar']            # (C, S)
    epochs = data['epochs']
    C, S = scalar.shape

    c, b, lam, success = fitmod.fit_all(scalar)
    fail_frac = float(np.mean(~success))

    figs = []
    figs.append(plots.plot_lambda_c_distributions(lam, c, success, output_dir))
    figs.append(plots.plot_example_curves(scalar, epochs, c, b, lam, output_dir))

    t, scalar_final, first_final, lam_ep = _designated_episode_timeline(data, c, lam)
    figs.append(plots.plot_episode_timeline(
        t, scalar_final, first_final, lam_ep, list(data['designated_timesteps']), output_dir))

    if npz_compare is not None:
        cmp = _load(npz_compare)
        c2, b2, lam2, _ = fitmod.fit_all(cmp['scalar'])
        t2, s2, _, l2 = _designated_episode_timeline(cmp, c2, lam2)
        # never overlay across abs_action (different action space)
        if bool(cmp['abs_action']) != bool(data['abs_action']):
            print("[analysis] WARNING: npz_compare has a different abs_action; "
                  "skipping cross-variant overlay (different action space).")
        else:
            figs.append(plots.plot_cross_variant_timeline(
                {'t': t, 'scalar_final': scalar_final, 'lam': lam_ep, 'label': str(data['obs_variant'])},
                {'t': t2, 'scalar_final': s2, 'lam': l2, 'label': str(cmp['obs_variant'])},
                output_dir))

    # ---- markdown summary ----
    md = os.path.join(output_dir, 'summary.md')
    lam_stats, c_stats = _q(lam), _q(c)
    lines = []
    lines.append(f"# Exp1 analysis — {data['task_name']} / {data['obs_variant']} / "
                 f"abs_action={bool(data['abs_action'])}\n")
    lines.append(f"- checkpoints: {C} (epochs {int(epochs[0])}..{int(epochs[-1])})")
    lines.append(f"- states: {S}  (H={int(data['horizon'])}, D={int(data['action_dim'])}, "
                 f"K_s={int(data['k_s'])}, M={int(data['ode_steps'])})")
    lines.append(f"- fit-failure fraction (fallback used): {fail_frac:.4f}\n")
    lines.append("## λ (rate) distribution")
    lines.append("".join([f"  {k}={v:.4g}" for k, v in lam_stats.items()]) + "\n")
    lines.append("## c (floor) distribution")
    lines.append("".join([f"  {k}={v:.4g}" for k, v in c_stats.items()]) + "\n")
    lines.append(f"## Designated states (episode {int(data['designated_episode_index'])})")
    lines.append("| frac | timestep | c (floor) | λ (rate) |")
    lines.append("|---|---|---|---|")
    for lbl, ts, pos in zip(data['designated_labels'], data['designated_timesteps'],
                            data['designated_positions']):
        lines.append(f"| {lbl} | {int(ts)} | {c[int(pos)]:.4g} | {lam[int(pos)]:.4g} |")
    lines.append("\n## Figures")
    for f in figs:
        lines.append(f"- {os.path.basename(f)}")
    with open(md, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[analysis] wrote {md} and {len(figs)} figures to {output_dir}")
    return md


@hydra.main(version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
    config_name='analysis')
def main(cfg):
    OmegaConf.resolve(cfg)
    analyze(
        npz_path=cfg.npz,
        output_dir=cfg.output_dir,
        npz_compare=cfg.get('npz_compare', None))


if __name__ == '__main__':
    main()
