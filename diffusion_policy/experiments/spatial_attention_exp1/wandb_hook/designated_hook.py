"""Designated-state MSE hook — the live wandb view.

Given one demonstration episode, this hook resolves four quarter-point env
timesteps, extracts the (obs, GT-action-chunk) for each ONCE at construction, then
on every rollout event computes MSE(o; theta) for those four states and returns a
flat dict of wandb scalars. The env-runner wrapper merges that dict into the
rollout log, so it is logged at the training loop's `rollout_every` cadence with no
core-loop edits.

Metric names (stable across tasks/episodes of different lengths):
  {prefix}/frac00, /frac25, /frac50, /frac75            (full chunk)
  {prefix}_first/frac00, ...                            (first executed action)
"""
import numpy as np

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric import state_provider
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager
from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.metric import PerStateMSEMetric


class DesignatedStateMSEHook:
    def __init__(self,
            dataset,
            n_obs_steps,
            episode_index=0,
            fractions=(0.0, 0.25, 0.5, 0.75),
            k_s=16,
            ode_steps=16,
            crn_seed=0,
            metric_prefix='designated_mse',
            max_eval_batch=256,
            num_fm=32,
            fm_seed=1):
        self.metric_prefix = metric_prefix
        self.episode_index = int(episode_index)
        self._summary_logged = False

        indices = dataset.sampler.indices
        episode_ends = dataset.replay_buffer.episode_ends

        last_t = state_provider.episode_last_timestep(episode_ends, self.episode_index)
        pairs = state_provider.resolve_quarter_timesteps(last_t, fractions)
        self.labels = [lbl for lbl, _ in pairs]
        self.timesteps = [t for _, t in pairs]

        sample_indices = [
            state_provider.resolve_sample_index(
                indices, episode_ends, n_obs_steps, self.episode_index, t, nearest=True)
            for t in self.timesteps]
        self.sample_indices = sample_indices
        # record the ACTUAL resolved timesteps (may differ from the ideal quarter
        # point for short episodes whose tail is unsampled)
        self.timesteps = [
            state_provider.invert_sample_index(indices, episode_ends, n_obs_steps, si)[1]
            for si in sample_indices]

        # extract the (few) designated states once, onto CPU
        self.obs_dict, self.gt = state_provider.build_state_batch(dataset, sample_indices)
        H, D = self.gt.shape[1], self.gt.shape[2]

        crn = CRNManager(k_s=k_s, horizon=H, action_dim=D,
            num_fm=num_fm, eps0_seed=crn_seed, fm_seed=fm_seed)
        self.metric = PerStateMSEMetric(crn, ode_steps=ode_steps, max_eval_batch=max_eval_batch)
        # drop the dataset reference so we don't pin a full (image) dataset in RAM
        del dataset

    def _log_summary_once(self):
        if self._summary_logged:
            return
        self._summary_logged = True
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.summary[f'{self.metric_prefix}/episode_index'] = self.episode_index
                for lbl, t, si in zip(self.labels, self.timesteps, self.sample_indices):
                    wandb.run.summary[f'{self.metric_prefix}/timestep_{lbl}'] = int(t)
                    wandb.run.summary[f'{self.metric_prefix}/sample_index_{lbl}'] = int(si)
        except Exception:
            pass  # wandb optional / offline

    def compute(self, policy):
        self._log_summary_once()
        res = self.metric.compute_mse(policy, self.obs_dict, self.gt)
        log = dict()
        for i, lbl in enumerate(self.labels):
            log[f'{self.metric_prefix}/{lbl}'] = float(res['scalar'][i])
            log[f'{self.metric_prefix}_first/{lbl}'] = float(res['first'][i])
        return log
