from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class FlowMatchingUnetLowdimPolicy(BaseLowdimPolicy):
    """
    Flow-matching (rectified-flow) counterpart of DiffusionUnetLowdimPolicy.

    Instead of a DDPM/DDIM noise scheduler, the generative model is a rectified
    flow with an independent coupling:
        x0 = clean (normalized) action trajectory (data)
        x1 = noise ~ N(0, I)
        x_t = (1 - t) * x0 + t * x1,     t ~ U(0, 1)   (t=0 -> data, t=1 -> noise)
    The network is trained to regress the constant velocity field
        v_theta(x_t, t, cond) ~= dx_t/dt = x1 - x0.
    Sampling integrates the ODE dx/dt = v_theta backward from t=1 (noise) to t=0
    (data) with a fixed number of Euler steps.

    The same ConditionalUnet1D backbone is reused unchanged; continuous time t is
    scaled by ``time_scale`` before being fed to the sinusoidal step embedding so
    it has useful resolution over [0, 1].
    """

    def __init__(self,
            model: ConditionalUnet1D,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=16,
            time_scale=1000.0,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.time_scale = time_scale
        self.num_inference_steps = num_inference_steps
        self.kwargs = kwargs

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        model = self.model
        device = condition_data.device
        dtype = condition_data.dtype
        B = condition_data.shape[0]

        # start from pure noise (t=1)
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=dtype,
            device=device,
            generator=generator)

        # fixed-step Euler integration of dx/dt = v_theta, from t=1 (noise) to t=0 (data)
        N = self.num_inference_steps
        dt = 1.0 / N
        for i in range(N):
            # current continuous time in [0, 1], decreasing from 1 towards 0
            t = 1.0 - i * dt

            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict velocity field (scale continuous time for the step embedding)
            t_tensor = torch.full((B,), t * self.time_scale, dtype=dtype, device=device)
            v = model(trajectory, t_tensor,
                local_cond=local_cond, global_cond=global_cond)

            # 3. Euler step backward in time: x_{t-dt} = x_t - v * dt
            trajectory = trajectory - v * dt

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise (x1) and a continuous flow time for each trajectory
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # t ~ U(0, 1), one per sample, clamped away from the endpoints
        t = torch.rand(bsz, device=trajectory.device).clamp(1e-4, 1 - 1e-4)
        # broadcast t over the (T, C) dims of the trajectory
        t_expand = t.view(bsz, *([1] * (trajectory.dim() - 1)))

        # rectified-flow interpolation: x_t = (1-t) * x0 + t * x1
        noisy_trajectory = (1.0 - t_expand) * trajectory + t_expand * noise
        # velocity target: dx_t/dt = x1 - x0
        target = noise - trajectory

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the velocity field (scale continuous time for the step embedding)
        pred = self.model(noisy_trajectory, t * self.time_scale,
            local_cond=local_cond, global_cond=global_cond)

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
