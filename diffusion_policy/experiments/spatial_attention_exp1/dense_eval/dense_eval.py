"""Offline dense evaluation: sweep every saved checkpoint of one run, computing the
per-state MSE arrays (scalar / per-index / first-executed) and the secondary FM loss
over a FIXED evaluation state set, and save everything (+ metadata) to an npz.

The evaluation state set, CRN, ODE steps, and designated states are all derived from
the checkpoint's own saved training cfg (``payload['cfg']``, which carries the full
``task.dataset`` and the ``exp1`` node), so the dense stage is exactly aligned with
training and needs no separate task config.
"""
import os
import re
import glob

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric import state_provider
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.metric import PerStateMSEMetric


def find_checkpoints(run_dir):
    """Return [(epoch, path), ...] sorted by epoch for epoch=*.ckpt in run_dir/checkpoints."""
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    paths = glob.glob(os.path.join(ckpt_dir, 'epoch=*.ckpt'))
    out = []
    for p in paths:
        m = re.search(r'epoch=(\d+)', os.path.basename(p))
        if m is not None:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _load_payload(path):
    with open(path, 'rb') as f:
        return torch.load(f, pickle_module=dill, map_location='cpu')


def _obs_variant(obs_dict):
    return 'lowdim' if set(obs_dict.keys()) == {'obs'} else 'image'


def run(run_dir, output_dir, device='cuda:0',
        subsample_size=None, subsample_seed=None, max_eval_batch=None,
        output_name='dense_eval.npz'):
    os.makedirs(output_dir, exist_ok=True)
    ckpts = find_checkpoints(run_dir)
    if len(ckpts) == 0:
        raise FileNotFoundError(
            f"No 'epoch=*.ckpt' checkpoints under {run_dir}/checkpoints. Train with "
            f"the exp1 config (checkpoint.topk.format_str='epoch={{epoch:04d}}.ckpt').")
    epochs = np.array([e for e, _ in ckpts], dtype=np.int64)

    # ---- reconstruct workspace once from the first checkpoint's cfg ----
    payload0 = _load_payload(ckpts[0][1])
    cfg = payload0['cfg']
    exp1 = cfg.get('exp1', OmegaConf.create({}))

    def pick(override, key, default):
        if override is not None:
            return override
        return exp1.get(key, default) if exp1 is not None else default

    subsample_size = pick(subsample_size, 'subsample_size', 2000)
    subsample_seed = pick(subsample_seed, 'subsample_seed', 0)
    max_eval_batch = pick(max_eval_batch, 'max_eval_batch', 256)
    k_s = int(exp1.get('k_s', 16))
    ode_steps = int(exp1.get('ode_steps', 16))
    crn_seed = int(exp1.get('crn_seed', 0))
    fm_seed = int(exp1.get('fm_crn_seed', 1))
    num_fm = int(exp1.get('fm_loss_samples', 32))
    episode_index = int(exp1.get('designated_episode_index', 0))
    fractions = list(exp1.get('quarter_fractions', [0.0, 0.25, 0.5, 0.75]))

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload0)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    n_obs_steps = int(policy.n_obs_steps)

    # ---- build the DP dataset exactly as trained (alignment) ----
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    indices = dataset.sampler.indices
    episode_ends = dataset.replay_buffer.episode_ends
    n = len(dataset)

    # ---- designated states (must be included) ----
    last_t = state_provider.episode_last_timestep(episode_ends, episode_index)
    pairs = state_provider.resolve_quarter_timesteps(last_t, fractions)
    designated_labels = [lbl for lbl, _ in pairs]
    designated_timesteps = [t for _, t in pairs]
    designated_indices = [
        state_provider.resolve_sample_index(
            indices, episode_ends, n_obs_steps, episode_index, t, nearest=True)
        for t in designated_timesteps]
    # actual resolved timesteps (snapped for short-episode tails)
    designated_timesteps = [
        state_provider.invert_sample_index(indices, episode_ends, n_obs_steps, si)[1]
        for si in designated_indices]

    # ---- fixed dense-eval state set (subsample ∪ full designated episode) ----
    # The full designated episode is included so the episode-timeline figure is
    # dense over the whole episode; the 4 quarter points are a subset of it.
    full_episode_indices = state_provider.episode_sample_indices(
        indices, episode_ends, n_obs_steps, episode_index)
    must_include = np.unique(np.concatenate([
        np.asarray(designated_indices, dtype=np.int64), full_episode_indices]))
    state_indices = state_provider.build_subsample(
        n, subsample_size, subsample_seed, must_include=must_include)
    designated_positions = [int(np.nonzero(state_indices == di)[0][0]) for di in designated_indices]

    # per-state (episode, timestep) metadata
    state_ep = np.empty(len(state_indices), dtype=np.int64)
    state_t = np.empty(len(state_indices), dtype=np.int64)
    for pos, si in enumerate(state_indices):
        e, t = state_provider.invert_sample_index(indices, episode_ends, n_obs_steps, si)
        state_ep[pos], state_t[pos] = e, t

    obs_dict, gt = state_provider.build_state_batch(dataset, state_indices)
    del dataset  # free (esp. image) after extracting states
    S, H, D = gt.shape
    obs_variant = _obs_variant(obs_dict)

    crn = CRNManager(k_s=k_s, horizon=H, action_dim=D,
        num_fm=num_fm, eps0_seed=crn_seed, fm_seed=fm_seed)
    metric = PerStateMSEMetric(crn, ode_steps=ode_steps, max_eval_batch=int(max_eval_batch))

    C = len(ckpts)
    scalar = np.empty((C, S), dtype=np.float64)
    per_index = np.empty((C, S, H), dtype=np.float64)
    first = np.empty((C, S), dtype=np.float64)
    fm_loss = np.empty((C, S), dtype=np.float64)

    for ci, (epoch, path) in enumerate(ckpts):
        payload = _load_payload(path) if ci > 0 else payload0
        workspace.load_payload(payload)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to(device)
        policy.eval()
        res = metric.compute_mse(policy, obs_dict, gt)
        scalar[ci] = res['scalar']
        per_index[ci] = res['per_index']
        first[ci] = res['first']
        fm_loss[ci] = metric.compute_fm_loss(policy, obs_dict, gt)
        print(f"[dense_eval] ckpt {ci+1}/{C} epoch={epoch} "
              f"mean_scalar_mse={scalar[ci].mean():.6f}")

    out_path = os.path.join(output_dir, output_name)
    np.savez(
        out_path,
        # arrays
        epochs=epochs,
        scalar=scalar,
        per_index=per_index,
        first=first,
        fm_loss=fm_loss,
        # state metadata
        state_indices=np.asarray(state_indices, dtype=np.int64),
        state_episode=state_ep,
        state_timestep=state_t,
        # designated metadata
        designated_labels=np.asarray(designated_labels, dtype=object),
        designated_timesteps=np.asarray(designated_timesteps, dtype=np.int64),
        designated_positions=np.asarray(designated_positions, dtype=np.int64),
        designated_episode_index=np.int64(episode_index),
        # variant identifiers
        task_name=str(cfg.task.name),
        obs_variant=str(obs_variant),
        abs_action=bool(cfg.task.get('abs_action', False)),
        # config echo
        horizon=np.int64(H),
        action_dim=np.int64(D),
        k_s=np.int64(k_s),
        ode_steps=np.int64(ode_steps),
        num_fm=np.int64(num_fm),
        crn_eps0_seed=np.int64(crn_seed),
        crn_fm_seed=np.int64(fm_seed),
        subsample_seed=np.int64(subsample_seed),
        subsample_size=np.int64(subsample_size if subsample_size is not None else n),
    )
    print(f"[dense_eval] wrote {out_path}  (C={C} checkpoints, S={S} states, H={H}, D={D})")
    return out_path
