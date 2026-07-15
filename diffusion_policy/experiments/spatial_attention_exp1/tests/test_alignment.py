"""Alignment tests against the REAL SequenceSampler / ReplayBuffer, so the
(episode, timestep) <-> dataset-index mapping and the executed-slice convention are
verified against repo behaviour rather than a re-derivation."""
import numpy as np
import pytest

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric import state_provider as sp

TO = 2          # n_obs_steps
H = 4           # horizon
N_ACTION = 2
EP_LENGTHS = [5, 7]


def _build():
    rb = ReplayBuffer.create_empty_numpy()
    for L in EP_LENGTHS:
        rb.add_episode({
            'obs': np.arange(L * 2).reshape(L, 2).astype(np.float32),
            'action': np.arange(L * 3).reshape(L, 3).astype(np.float32),
        })
    sampler = SequenceSampler(
        rb, sequence_length=H, pad_before=TO - 1, pad_after=N_ACTION - 1)
    return rb, sampler


def test_resolve_and_invert_roundtrip():
    rb, sampler = _build()
    indices = sampler.indices
    ends = rb.episode_ends
    ep_starts = np.concatenate([[0], ends[:-1]])
    for e, L in enumerate(EP_LENGTHS):
        # pad_after = N_ACTION-1 leaves the episode tail unsampled: current-obs
        # timesteps run 0..L-2 for these params.
        for t in range(L - 1):
            i = sp.resolve_sample_index(indices, ends, TO, e, t)
            # the current-obs window slot (index TO-1) must hold env timestep t
            sample = sampler.sample_sequence(i)
            abs_step = int(ep_starts[e]) + t
            assert np.allclose(sample['obs'][TO - 1], rb['obs'][abs_step])
            # inverse recovers (e, t)
            assert sp.invert_sample_index(indices, ends, TO, i) == (e, t)


def test_nearest_snaps_unsampled_tail():
    rb, sampler = _build()
    indices, ends = sampler.indices, rb.episode_ends
    L = EP_LENGTHS[0]
    # exact resolution of the last timestep fails (unsampled tail) ...
    import pytest
    with pytest.raises(ValueError):
        sp.resolve_sample_index(indices, ends, TO, 0, L - 1)
    # ... but nearest snaps to the last available sample (timestep L-2)
    i = sp.resolve_sample_index(indices, ends, TO, 0, L - 1, nearest=True)
    assert sp.invert_sample_index(indices, ends, TO, i) == (0, L - 2)


def test_executed_slice_gt_alignment():
    # first executed action aligns to chunk index start = TO-1; GT there is the
    # action at env timestep t (matches predict_action / workspace slicing).
    rb, sampler = _build()
    indices, ends = sampler.indices, rb.episode_ends
    ep_starts = np.concatenate([[0], ends[:-1]])
    e, t = 1, 3
    i = sp.resolve_sample_index(indices, ends, TO, e, t)
    sample = sampler.sample_sequence(i)
    start = TO - 1
    abs_step = int(ep_starts[e]) + t
    assert np.allclose(sample['action'][start], rb['action'][abs_step])


def test_episode_sample_indices_cover_episode():
    rb, sampler = _build()
    indices, ends = sampler.indices, rb.episode_ends
    for e, L in enumerate(EP_LENGTHS):
        idxs = sp.episode_sample_indices(indices, ends, TO, e)
        # current-obs timesteps 0..L-2 are sampled (tail unsampled by pad_after)
        assert len(idxs) == L - 1
        recovered_t = sorted(sp.invert_sample_index(indices, ends, TO, i)[1] for i in idxs)
        assert recovered_t == list(range(L - 1))


def test_missing_timestep_raises():
    rb, sampler = _build()
    indices, ends = sampler.indices, rb.episode_ends
    with pytest.raises(ValueError):
        sp.resolve_sample_index(indices, ends, TO, 0, 999)
