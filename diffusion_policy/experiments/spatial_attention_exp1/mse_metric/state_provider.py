"""Variant-agnostic dataset-state plumbing for the MSE experiment.

A "state" is a deterministic dataset sample ``dataset[i]`` (the SequenceSampler
read path has no RNG). This module maps between (episode, environment-timestep)
and dataset indices, resolves the designated quarter-point timesteps, batches
samples into the (obs_dict, gt) tensors the metric consumes, and builds the fixed
dense-eval subsample.

Key alignment facts (verified against diffusion_policy/common/sampler.py):
  * For sample row ``r`` with ``indices[r] = [buf_start, buf_end, samp_start, samp_end]``,
    the env timestep sitting at horizon-window index ``To-1`` (the policy's "current"
    observation) is ``current_abs = buf_start + (To-1) - samp_start`` in the
    concatenated replay buffer.
  * The GT action chunk the policy is scored on is ``sample['action']`` (full
    horizon); the executed slice / first-executed action aligns to chunk index
    ``start = To-1``.
"""
import math
import numpy as np
import torch


def resolve_quarter_timesteps(last_timestep: int, fractions):
    """Resolve designated env timesteps for one episode.

    ``last_timestep`` = T (the episode's last environment timestep, i.e.
    episode_length - 1). Returns a list of ``(frac_label, timestep)`` pairs, ONE
    per requested fraction (not de-duplicated — the wandb hook logs a curve per
    fraction). ``timestep = clamp(floor(frac * T), 0, T)``. Short episodes may
    collapse several fractions onto the same timestep; that is expected.
    """
    T = int(last_timestep)
    out = []
    for f in fractions:
        t = int(math.floor(float(f) * T))
        t = max(0, min(T, t))
        label = f"frac{int(round(float(f) * 100)):02d}"
        out.append((label, t))
    return out


def resolve_sample_index(indices, episode_ends, n_obs_steps, episode, timestep, nearest=False):
    """Pure mapping (episode, timestep) -> dataset sample index.

    ``indices`` is ``dataset.sampler.indices`` (N x 4). ``episode_ends`` is
    ``replay_buffer.episode_ends``. Because ``pad_after = n_action_steps-1``, an
    episode of length L only has samples whose current observation sits at env
    timesteps roughly ``0..L-2``; the last timesteps have no sample. With
    ``nearest=False`` (default, exact) a missing timestep raises ValueError; with
    ``nearest=True`` the closest existing sample WITHIN the episode is returned
    (used for designated-state resolution so short episodes still resolve). Raises
    if the episode contributes no samples at all (e.g. a held-out val episode).
    """
    indices = np.asarray(indices)
    episode_ends = np.asarray(episode_ends)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    if episode < 0 or episode >= len(episode_ends):
        raise ValueError(f"episode {episode} out of range [0, {len(episode_ends)})")
    target_abs = int(ep_starts[episode]) + int(timestep)
    current_abs = indices[:, 0] + (int(n_obs_steps) - 1) - indices[:, 2]
    if nearest:
        in_ep = (current_abs >= int(ep_starts[episode])) & (current_abs < int(episode_ends[episode]))
        rows = np.nonzero(in_ep)[0]
        if len(rows) == 0:
            raise ValueError(
                f"Episode {episode} contributes no dataset samples (val split?).")
        j = int(np.argmin(np.abs(current_abs[rows] - target_abs)))
        return int(rows[j])
    matches = np.nonzero(current_abs == target_abs)[0]
    if len(matches) == 0:
        raise ValueError(
            f"No dataset sample for (episode={episode}, timestep={timestep}). "
            f"The episode may be absent from the train sampler (val split), the "
            f"timestep is out of range, or it falls in the unsampled tail "
            f"(last ~n_action_steps steps). Use nearest=True to snap.")
    return int(matches[0])


def invert_sample_index(indices, episode_ends, n_obs_steps, i):
    """Recover (episode, timestep) for dataset sample index ``i`` (for metadata)."""
    indices = np.asarray(indices)
    episode_ends = np.asarray(episode_ends)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    current_abs = int(indices[i, 0]) + (int(n_obs_steps) - 1) - int(indices[i, 2])
    episode = int(np.searchsorted(episode_ends, current_abs, side='right'))
    episode = min(episode, len(episode_ends) - 1)
    timestep = current_abs - int(ep_starts[episode])
    return episode, timestep


def episode_last_timestep(episode_ends, episode):
    """Last env timestep index (episode_length - 1) for ``episode``."""
    episode_ends = np.asarray(episode_ends)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    length = int(episode_ends[episode]) - int(ep_starts[episode])
    return length - 1


def episode_sample_indices(indices, episode_ends, n_obs_steps, episode):
    """All dataset sample indices whose current-observation timestep falls in
    ``episode`` (used to densely cover the designated episode's timeline)."""
    indices = np.asarray(indices)
    episode_ends = np.asarray(episode_ends)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    current_abs = indices[:, 0] + (int(n_obs_steps) - 1) - indices[:, 2]
    mask = (current_abs >= int(ep_starts[episode])) & (current_abs < int(episode_ends[episode]))
    return np.nonzero(mask)[0].astype(np.int64)


def build_subsample(n, size, seed, must_include=None):
    """Fixed random subsample of ``range(n)`` (size ``size``), unioned with
    ``must_include``. Deterministic given ``seed``. Returns a sorted int array."""
    rng = np.random.default_rng(int(seed))
    if size is None or size >= n:
        base = np.arange(n)
    else:
        base = rng.choice(n, size=int(size), replace=False)
    if must_include is not None and len(must_include) > 0:
        base = np.concatenate([base, np.asarray(must_include, dtype=np.int64)])
    return np.unique(base.astype(np.int64))


def sample_obs_is_image(sample):
    """True if the dataset sample carries an image (dict-of-modalities) obs."""
    return isinstance(sample['obs'], dict)


def build_state_batch(dataset, sample_indices):
    """Stack ``dataset[i]`` for ``i in sample_indices`` into batched tensors.

    Returns ``(obs_dict, gt)`` where:
      * lowdim: ``obs_dict = {'obs': (S, H, obs_dim)}``
      * image:  ``obs_dict = {modality: (S, To, ...)}``
      * ``gt``: ``(S, H, action_dim)`` (raw / unnormalized action chunk)
    All tensors are CPU float; move to device at use time.
    """
    obs_list = []
    gt_list = []
    is_image = None
    for i in sample_indices:
        sample = dataset[int(i)]
        if is_image is None:
            is_image = sample_obs_is_image(sample)
        gt_list.append(sample['action'].float())
        if is_image:
            obs_list.append({k: v.float() for k, v in sample['obs'].items()})
        else:
            obs_list.append(sample['obs'].float())

    gt = torch.stack(gt_list, dim=0)
    if is_image:
        keys = obs_list[0].keys()
        obs_dict = {k: torch.stack([o[k] for o in obs_list], dim=0) for k in keys}
    else:
        obs_dict = {'obs': torch.stack(obs_list, dim=0)}
    return obs_dict, gt


def slice_state_batch(obs_dict, gt, start, end):
    """Slice a batched (obs_dict, gt) along the state dimension."""
    obs_chunk = {k: v[start:end] for k, v in obs_dict.items()}
    return obs_chunk, gt[start:end]


def num_states(gt):
    return gt.shape[0]
