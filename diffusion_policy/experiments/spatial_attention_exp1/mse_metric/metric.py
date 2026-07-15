"""Per-state sampled-action MSE metric (variant-agnostic).

Routes entirely through ``policy.predict_action`` so it is identical for low_dim
and image variants — it never touches observations directly. It injects the fixed
CRN initial noise via ``policy._init_noise`` (see the Exp1 policy subclasses) and
forces a fixed number of ODE steps ``M`` so every evaluation is comparable.

Normalization convention (documented decision): to match the repo's existing
``train_action_mse_error`` (which is ``F.mse_loss`` — a mean over all elements), all
MSE variants are means over their non-batch dims (mean over K_s always):
  * ``scalar``    : mean over (K_s, H, D)
  * ``per_index`` : mean over (K_s, D) -> one value per chunk index k
  * ``first``     : mean over (K_s, D) at chunk index start = To-1 (the executed slice)
All values are in RAW (unnormalized) action space (predict_action unnormalizes).
TODO: a sum-of-squares (``||.||^2``) variant is a one-line change if ever needed.
"""
import contextlib
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric import state_provider


def _executed_start(policy):
    """Chunk index of the first executed action (matches predict_action slicing)."""
    start = policy.n_obs_steps
    if getattr(policy, 'oa_step_convention', True):
        start = policy.n_obs_steps - 1
    return int(start)


@contextlib.contextmanager
def _forced_ode_steps(policy, m):
    """Temporarily force policy.num_inference_steps = M (restored on exit)."""
    prev = policy.num_inference_steps
    policy.num_inference_steps = int(m)
    try:
        yield
    finally:
        policy.num_inference_steps = prev


@contextlib.contextmanager
def _injected_noise(policy, noise):
    prev = getattr(policy, '_init_noise', None)
    policy._init_noise = noise
    try:
        yield
    finally:
        policy._init_noise = prev


class PerStateMSEMetric:
    def __init__(self, crn, ode_steps: int, max_eval_batch: int = 256):
        self.crn = crn
        self.ode_steps = int(ode_steps)
        self.max_eval_batch = int(max_eval_batch)

    @torch.no_grad()
    def compute_mse(self, policy, obs_dict, gt):
        """Per-state MSE via K_s CRN samples through the full ODE sampler.

        obs_dict/gt are batched over S states (CPU tensors ok). Returns a dict of
        numpy arrays: ``scalar (S,)``, ``per_index (S, H)``, ``first (S,)``.
        """
        assert hasattr(policy, '_init_noise'), (
            "policy must be an Exp1 flow-matching policy exposing `_init_noise`; "
            "otherwise the CRN noise cannot be injected and MSE would use fresh "
            "noise every call.")
        device = policy.device
        K = self.crn.k_s
        S = state_provider.num_states(gt)
        H = self.crn.horizon
        start = _executed_start(policy)

        eps0 = self.crn.eps0.to(device=device, dtype=policy.dtype)  # (K, H, D)

        scalar = np.empty(S, dtype=np.float64)
        per_index = np.empty((S, H), dtype=np.float64)
        first = np.empty(S, dtype=np.float64)

        # chunk states so that (chunk_S * K) <= max_eval_batch
        chunk_S = max(1, self.max_eval_batch // K)
        for s0 in range(0, S, chunk_S):
            s1 = min(S, s0 + chunk_S)
            cs = s1 - s0
            obs_chunk, gt_chunk = state_provider.slice_state_batch(obs_dict, gt, s0, s1)
            obs_rep = dict_apply(
                obs_chunk,
                lambda x: x.repeat_interleave(K, dim=0).to(device))    # (cs*K, ...)
            gt_dev = gt_chunk.to(device)                                # (cs, H, D)
            # tile CRN noise: row s*K+j uses eps0[j] (matches obs repeat_interleave)
            noise = eps0.unsqueeze(0).expand(cs, *eps0.shape).reshape(cs * K, *eps0.shape[1:])

            with _forced_ode_steps(policy, self.ode_steps), _injected_noise(policy, noise):
                result = policy.predict_action(obs_rep)
            pred = result['action_pred'].reshape(cs, K, H, -1)          # (cs, K, H, D)

            diff2 = (pred - gt_dev.unsqueeze(1)) ** 2                    # (cs, K, H, D)
            scalar[s0:s1] = diff2.mean(dim=(1, 2, 3)).double().cpu().numpy()
            per_index[s0:s1] = diff2.mean(dim=(1, 3)).double().cpu().numpy()
            first[s0:s1] = diff2[:, :, start, :].mean(dim=(1, 2)).double().cpu().numpy()

        return {'scalar': scalar, 'per_index': per_index, 'first': first}

    @torch.no_grad()
    def compute_fm_loss(self, policy, obs_dict, gt):
        """Secondary per-state FM (velocity-regression) loss over the fixed
        {(tau_j, eps_j)} set. Returns numpy array ``(S,)``. Storage only."""
        device = policy.device
        S = state_provider.num_states(gt)
        taus = self.crn.taus.to(device=device, dtype=policy.dtype)        # (num_fm,)
        eps_fm = self.crn.eps_fm.to(device=device, dtype=policy.dtype)    # (num_fm, H, D)
        num_fm = self.crn.num_fm
        time_scale = getattr(policy, 'time_scale', 1.0)

        out = np.empty(S, dtype=np.float64)
        chunk_S = max(1, self.max_eval_batch)
        for s0 in range(0, S, chunk_S):
            s1 = min(S, s0 + chunk_S)
            cs = s1 - s0
            obs_chunk, gt_chunk = state_provider.slice_state_batch(obs_dict, gt, s0, s1)
            obs_chunk = dict_apply(obs_chunk, lambda x: x.to(device))
            gt_dev = gt_chunk.to(device)                                  # (cs, H, D)
            gc = policy.fm_global_cond(obs_chunk)                         # (cs, gdim) — encode once

            acc = torch.zeros(cs, device=device, dtype=torch.float64)
            for j in range(num_fm):
                tau = taus[j]
                eps = eps_fm[j].unsqueeze(0)                              # (1, H, D)
                x = (1.0 - tau) * gt_dev + tau * eps                      # (cs, H, D)
                t_tensor = torch.full((cs,), float(tau) * time_scale,
                    device=device, dtype=policy.dtype)
                pred_v = policy.model(x, t_tensor, global_cond=gc)        # (cs, H, D)
                target = eps - gt_dev
                acc += ((pred_v - target) ** 2).mean(dim=(1, 2)).double()
            out[s0:s1] = (acc / num_fm).cpu().numpy()
        return out
