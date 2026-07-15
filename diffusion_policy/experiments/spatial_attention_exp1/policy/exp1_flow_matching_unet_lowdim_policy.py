"""Experiment-1 subclass of the low-dim flow-matching policy.

Adds exactly two things on top of `FlowMatchingUnetLowdimPolicy`, without touching
its forward math:

  1. An *injectable initial-noise pathway* in `conditional_sample`: when the
     attribute `self._init_noise` is set (shape ``(B, horizon, action_dim)``), it
     replaces the internal `torch.randn` draw. This lets the MSE metric feed a
     FIXED common-random-number set {eps0_j} so the same noise is reused across
     states / rollout events / checkpoints / variants. When `_init_noise is None`
     the behaviour is identical to the parent (plain `torch.randn`).

  2. `fm_global_cond(obs_dict)` — exposes the same `global_cond` tensor the
     parent's `predict_action` builds, used only by the secondary FM-loss metric.

`predict_action` is inherited unchanged, so the metric injects noise simply by
setting `policy._init_noise` around a `predict_action` call.
"""
import torch

from diffusion_policy.policy.flow_matching_unet_lowdim_policy import (
    FlowMatchingUnetLowdimPolicy,
)


class Exp1FlowMatchingUnetLowdimPolicy(FlowMatchingUnetLowdimPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # transient CRN noise; set by the metric around predict_action, else None
        self._init_noise = None

    # ========= inference (CRN-injectable) ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs):
        model = self.model
        device = condition_data.device
        dtype = condition_data.dtype
        B = condition_data.shape[0]

        # start from noise (t=1); inject the fixed CRN noise when provided
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=dtype,
            device=device,
            generator=generator)
        init_noise = getattr(self, '_init_noise', None)
        if init_noise is not None:
            init_noise = init_noise.to(device=device, dtype=dtype)
            assert init_noise.shape[0] == B and init_noise.shape[1] == condition_data.shape[1], (
                f"_init_noise {tuple(init_noise.shape)} incompatible with "
                f"condition_data {tuple(condition_data.shape)}")
            da = init_noise.shape[-1]
            # global_cond path: da == last dim (overwrites all); impainting path:
            # only the first `da` (action) channels are CRN-controlled.
            trajectory[..., :da] = init_noise

        N = self.num_inference_steps
        dt = 1.0 / N
        for i in range(N):
            t = 1.0 - i * dt
            trajectory[condition_mask] = condition_data[condition_mask]
            t_tensor = torch.full((B,), t * self.time_scale, dtype=dtype, device=device)
            v = model(trajectory, t_tensor,
                local_cond=local_cond, global_cond=global_cond)
            trajectory = trajectory - v * dt

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ========= secondary FM-loss support ============
    def fm_global_cond(self, obs_dict):
        """Return the `global_cond` that `predict_action` would build for this obs,
        for the secondary per-state FM-loss metric. Assumes obs_as_global_cond."""
        # TODO: support obs_as_local_cond / impainting conditioning.
        assert self.obs_as_global_cond, \
            "fm_global_cond currently assumes obs_as_global_cond=True"
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        To = self.n_obs_steps
        return nobs[:, :To].reshape(nobs.shape[0], -1)
