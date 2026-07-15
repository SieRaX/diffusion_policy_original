"""Metric behaviour via a tiny dummy policy (no GPU / UNet / robomimic):
CRN reuse across states + calls, fixed M everywhere, and executed-slice alignment."""
import numpy as np
import torch

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.metric import PerStateMSEMetric


class _DummyModel:
    def __call__(self, x, t, global_cond=None):
        return torch.zeros_like(x)


class DummyPolicy:
    """Minimal stand-in exposing exactly what the metric touches."""
    def __init__(self, mode='noise', n_obs_steps=2, horizon=4, action_dim=3):
        self.mode = mode
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.action_dim = action_dim
        self.oa_step_convention = True
        self.num_inference_steps = 999      # default; the metric must force M
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        self.time_scale = 1000.0
        self.model = _DummyModel()
        self._init_noise = None
        self.seen_M = []

    def eval(self):
        return self

    def predict_action(self, obs_dict):
        self.seen_M.append(self.num_inference_steps)
        B = self._init_noise.shape[0]
        if self.mode == 'noise':
            ap = self._init_noise.clone()                       # (B, H, D) == tiled eps0
        else:  # 'zeros'
            ap = torch.zeros(B, self.horizon, self.action_dim)
        start = self.n_obs_steps - 1
        return {'action_pred': ap, 'action': ap[:, start:start + 1]}

    def fm_global_cond(self, obs_dict):
        B = next(iter(obs_dict.values())).shape[0]
        return torch.zeros(B, 4)


def _states(S, To, obs_dim, H, D):
    obs = {'obs': torch.zeros(S, To, obs_dim)}
    gt = torch.zeros(S, H, D)
    return obs, gt


def test_crn_reuse_across_states_and_calls():
    H, D, K = 4, 3, 8
    policy = DummyPolicy(mode='noise', horizon=H, action_dim=D)
    crn = CRNManager(k_s=K, horizon=H, action_dim=D, eps0_seed=0)
    metric = PerStateMSEMetric(crn, ode_steps=5, max_eval_batch=256)
    obs, gt = _states(6, policy.n_obs_steps, 5, H, D)

    r1 = metric.compute_mse(policy, obs, gt)
    r2 = metric.compute_mse(policy, obs, gt)
    # deterministic across calls (same CRN reused)
    assert np.allclose(r1['scalar'], r2['scalar'])
    # every state saw the SAME {eps0_j} -> identical scalar across states (gt=0)
    assert np.allclose(r1['scalar'], r1['scalar'][0])
    # value equals mean of eps0^2
    assert abs(r1['scalar'][0] - float((crn.eps0 ** 2).mean())) < 1e-6


def test_fixed_M_forced_and_restored():
    H, D, K = 4, 3, 4
    policy = DummyPolicy(mode='noise', horizon=H, action_dim=D)
    crn = CRNManager(k_s=K, horizon=H, action_dim=D)
    metric = PerStateMSEMetric(crn, ode_steps=7, max_eval_batch=64)
    obs, gt = _states(3, policy.n_obs_steps, 5, H, D)
    metric.compute_mse(policy, obs, gt)
    # M forced to 7 for every predict_action call, and restored afterwards
    assert set(policy.seen_M) == {7}
    assert policy.num_inference_steps == 999


def test_executed_slice_alignment():
    H, D, K = 5, 3, 2
    To = 2
    start = To - 1
    policy = DummyPolicy(mode='zeros', n_obs_steps=To, horizon=H, action_dim=D)
    crn = CRNManager(k_s=K, horizon=H, action_dim=D)
    metric = PerStateMSEMetric(crn, ode_steps=3, max_eval_batch=64)

    S = 3
    obs = {'obs': torch.zeros(S, To, 5)}
    gt = torch.zeros(S, H, D)
    gt[:, start, :] = 1.0            # spike only at the executed chunk index

    res = metric.compute_mse(policy, obs, gt)
    # per-index isolates the spike at `start`
    assert np.allclose(res['per_index'][:, start], 1.0)
    others = [k for k in range(H) if k != start]
    assert np.allclose(res['per_index'][:, others], 0.0)
    # first-executed MSE equals the spike; scalar = 1/H
    assert np.allclose(res['first'], 1.0)
    assert np.allclose(res['scalar'], 1.0 / H)


def test_fm_loss_deterministic():
    H, D = 4, 3
    policy = DummyPolicy(horizon=H, action_dim=D)
    crn = CRNManager(k_s=2, horizon=H, action_dim=D, num_fm=16, fm_seed=1)
    metric = PerStateMSEMetric(crn, ode_steps=3, max_eval_batch=64)
    obs, gt = _states(4, policy.n_obs_steps, 5, H, D)
    a = metric.compute_fm_loss(policy, obs, gt)
    b = metric.compute_fm_loss(policy, obs, gt)
    assert np.allclose(a, b)
    assert np.all(np.isfinite(a))
