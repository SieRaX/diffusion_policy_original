"""Integration test for DesignatedStateMSEHook: build from a (synthetic) dataset
and produce the wandb metric dict via a dummy policy — the same path the workspace
drives at sample_every cadence."""
import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.experiments.spatial_attention_exp1.wandb_hook.designated_hook import (
    DesignatedStateMSEHook,
)
from diffusion_policy.experiments.spatial_attention_exp1.tests.test_metric import DummyPolicy

TO, H, N_ACTION, OBS_DIM, ACT_DIM = 2, 4, 2, 5, 3


class FakeLowdimDataset:
    """Minimal stand-in exposing the attributes the hook uses."""
    def __init__(self, ep_lengths):
        rb = ReplayBuffer.create_empty_numpy()
        for L in ep_lengths:
            rb.add_episode({
                'obs': np.random.randn(L, OBS_DIM).astype(np.float32),
                'action': np.random.randn(L, ACT_DIM).astype(np.float32),
            })
        self.replay_buffer = rb
        self.sampler = SequenceSampler(
            rb, sequence_length=H, pad_before=TO - 1, pad_after=N_ACTION - 1)

    def __getitem__(self, i):
        d = self.sampler.sample_sequence(i)
        return {k: torch.from_numpy(v) for k, v in d.items()}


def test_hook_produces_designated_metrics():
    dataset = FakeLowdimDataset([20, 25])
    hook = DesignatedStateMSEHook(
        dataset=dataset, n_obs_steps=TO, episode_index=0,
        fractions=[0.0, 0.25, 0.5, 0.75], k_s=4, ode_steps=3,
        crn_seed=0, metric_prefix='designated_mse', max_eval_batch=32)

    policy = DummyPolicy(mode='noise', n_obs_steps=TO, horizon=H, action_dim=ACT_DIM)
    log = hook.compute(policy)

    # one full-chunk + one first-executed metric per fraction
    for frac in ['frac00', 'frac25', 'frac50', 'frac75']:
        assert f'designated_mse/{frac}' in log
        assert f'designated_mse_first/{frac}' in log
    assert all(np.isfinite(v) for v in log.values())
    assert len(log) == 8


def test_hook_crn_reused_across_calls():
    dataset = FakeLowdimDataset([20, 25])
    hook = DesignatedStateMSEHook(
        dataset=dataset, n_obs_steps=TO, episode_index=0,
        fractions=[0.0, 0.25, 0.5, 0.75], k_s=4, ode_steps=3,
        crn_seed=0, metric_prefix='designated_mse', max_eval_batch=32)
    policy = DummyPolicy(mode='noise', n_obs_steps=TO, horizon=H, action_dim=ACT_DIM)
    a = hook.compute(policy)
    b = hook.compute(policy)
    # deterministic across rollout events (same CRN reused)
    assert a == b


class FakeImageDataset:
    """Image-style dataset: obs is a dict of modalities, truncated to n_obs_steps
    (as RobomimicReplayImageDataset does); action keeps the full horizon."""
    def __init__(self, ep_lengths):
        rb = ReplayBuffer.create_empty_numpy()
        for L in ep_lengths:
            rb.add_episode({
                'cam': np.random.rand(L, 3, 8, 8).astype(np.float32),
                'pos': np.random.randn(L, 4).astype(np.float32),
                'action': np.random.randn(L, ACT_DIM).astype(np.float32),
            })
        self.replay_buffer = rb
        self.sampler = SequenceSampler(
            rb, sequence_length=H, pad_before=TO - 1, pad_after=N_ACTION - 1)

    def __getitem__(self, i):
        d = self.sampler.sample_sequence(i)
        obs = {
            'cam': torch.from_numpy(d['cam'][:TO]).float(),
            'pos': torch.from_numpy(d['pos'][:TO]).float(),
        }
        return {'obs': obs, 'action': torch.from_numpy(d['action']).float()}


def test_hook_compatible_with_image_dict_obs():
    dataset = FakeImageDataset([20, 25])
    hook = DesignatedStateMSEHook(
        dataset=dataset, n_obs_steps=TO, episode_index=0,
        fractions=[0.0, 0.25, 0.5, 0.75], k_s=4, ode_steps=3,
        crn_seed=0, metric_prefix='designated_mse', max_eval_batch=32)
    policy = DummyPolicy(mode='noise', n_obs_steps=TO, horizon=H, action_dim=ACT_DIM)
    log = hook.compute(policy)
    for frac in ['frac00', 'frac25', 'frac50', 'frac75']:
        assert f'designated_mse/{frac}' in log
        assert f'designated_mse_first/{frac}' in log
    assert all(np.isfinite(v) for v in log.values())


def test_hook_drops_dataset_reference():
    dataset = FakeLowdimDataset([20])
    hook = DesignatedStateMSEHook(
        dataset=dataset, n_obs_steps=TO, episode_index=0,
        fractions=[0.0, 0.25, 0.5, 0.75], k_s=2, ode_steps=2,
        crn_seed=0, metric_prefix='designated_mse', max_eval_batch=16)
    # the hook must not retain the (potentially large) dataset
    assert not hasattr(hook, 'dataset')
