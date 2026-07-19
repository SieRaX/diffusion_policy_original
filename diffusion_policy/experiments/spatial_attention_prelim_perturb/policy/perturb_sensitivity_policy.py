"""Policy subclasses that expose the PRE-unnormalize (normalized) action chunk
alongside the raw one, while keeping the Exp1 CRN `_init_noise` injection.

Rather than duplicate the parent `predict_action` body, we wrap `conditional_sample`
to stash the raw sampler output (`nsample`) on the instance, then call the parent
`predict_action` and read the normalized chunk `naction_pred = nsample[..., :Da]`
from the stash. `predict_action`'s result gains:
  - `naction_pred` : (B, horizon, Da) normalized (before unnormalize)
  - `naction`      : (B, n_action_steps, Da) normalized executed slice
and asserts `unnormalize(naction_pred) == action_pred` (the two must be bitwise
consistent — they are, since the parent computes action_pred = unnormalize(naction_pred)).
"""
import torch

from diffusion_policy.experiments.spatial_attention_exp1.policy.exp1_flow_matching_unet_lowdim_policy import (
    Exp1FlowMatchingUnetLowdimPolicy,
)
from diffusion_policy.experiments.spatial_attention_exp1.policy.exp1_flow_matching_unet_hybrid_image_policy import (
    Exp1FlowMatchingUnetHybridImagePolicy,
)
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.metric import _executed_start


class _NormalizedChunkMixin:
    """Adds normalized-chunk exposure on top of an Exp1 FM policy subclass."""

    def conditional_sample(self, *args, **kwargs):
        traj = super().conditional_sample(*args, **kwargs)
        # stash the raw sampler trajectory so predict_action can read the
        # normalized (pre-unnormalize) chunk from it
        self._last_trajectory = traj
        return traj

    def predict_action(self, obs_dict):
        result = super().predict_action(obs_dict)  # runs self.conditional_sample -> stash
        Da = self.action_dim
        nsample = self._last_trajectory
        naction_pred = nsample[..., :Da]

        # executed slice in normalized space (same alignment as raw `action`)
        start = _executed_start(self)
        end = start + self.n_action_steps
        naction = naction_pred[:, start:end]

        # raw and normalized chunks must be bitwise-consistent via the affine normalizer
        assert torch.allclose(
            self.normalizer['action'].unnormalize(naction_pred),
            result['action_pred'], atol=1e-5, rtol=1e-4), \
            "unnormalize(naction_pred) != action_pred"

        result['naction_pred'] = naction_pred
        result['naction'] = naction
        return result


class PerturbSensitivityLowdimPolicy(_NormalizedChunkMixin, Exp1FlowMatchingUnetLowdimPolicy):
    pass


class PerturbSensitivityHybridImagePolicy(_NormalizedChunkMixin, Exp1FlowMatchingUnetHybridImagePolicy):
    pass
