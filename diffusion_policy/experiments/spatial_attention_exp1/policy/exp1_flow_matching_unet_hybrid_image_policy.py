"""Experiment-1 subclass of the hybrid-image flow-matching policy.

Identical additions to the low-dim Exp1 policy (see that file's docstring):
an injectable `self._init_noise` pathway in `conditional_sample` for CRN, and a
`fm_global_cond` helper (which here runs the visual obs-encoder) for the secondary
FM-loss metric. The parent's forward math and `predict_action` are untouched.
"""
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.flow_matching_unet_hybrid_image_policy import (
    FlowMatchingUnetHybridImagePolicy,
)


class Exp1FlowMatchingUnetHybridImagePolicy(FlowMatchingUnetHybridImagePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        """Encode obs into the `global_cond` used by predict_action. Assumes
        obs_as_global_cond."""
        # TODO: support the impainting (obs_as_global_cond=False) path.
        assert self.obs_as_global_cond, \
            "fm_global_cond currently assumes obs_as_global_cond=True"
        nobs = self.normalizer.normalize(obs_dict)
        To = self.n_obs_steps
        B = next(iter(obs_dict.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, -1)
