"""Designated-state MSE hook — the live wandb view.

Given one demonstration episode, this hook:
  (a) resolves four quarter-point env timesteps and, on every call, logs the
      per-state MSE for each as wandb SCALARS (x-axis = training step):
        {prefix}/frac00 .. /frac75            (full chunk)
        {prefix}_first/frac00 ..              (first executed action)
  (b) optionally evaluates MSE across the WHOLE designated episode and logs a
      line plot (x = episode rollout timestep, y = MSE) as a wandb.Image under
        {prefix}_episode_curve
      Logging an image every cadence makes wandb render a media panel with a
      training-step slider ("controllable bar") to scrub how the curve evolves.

All (obs, GT-action-chunk) states are extracted ONCE at construction (then the
dataset reference is dropped). The metric uses common random numbers, so the same
noise is reused across states, calls, and checkpoints.
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
            fm_seed=1,
            episode_curve=True,
            episode_curve_stride=1,
            episode_curve_every=1):
        self.metric_prefix = metric_prefix
        self.episode_index = int(episode_index)
        self.episode_curve = bool(episode_curve)
        self.episode_curve_every = max(1, int(episode_curve_every))
        self._summary_logged = False
        self._curve_call_count = 0

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

        # optionally extract the FULL designated episode (sorted by timestep) for
        # the episode-curve image
        self.ep_obs_dict = None
        self.ep_gt = None
        self.ep_timesteps = None
        if self.episode_curve:
            ep_idx = state_provider.episode_sample_indices(
                indices, episode_ends, n_obs_steps, self.episode_index)
            ep_ts = np.array([
                state_provider.invert_sample_index(indices, episode_ends, n_obs_steps, i)[1]
                for i in ep_idx])
            order = np.argsort(ep_ts)
            ep_idx, ep_ts = ep_idx[order], ep_ts[order]
            stride = max(1, int(episode_curve_stride))
            ep_idx, ep_ts = ep_idx[::stride], ep_ts[::stride]
            self.ep_timesteps = ep_ts
            self.ep_obs_dict, self.ep_gt = state_provider.build_state_batch(
                dataset, ep_idx.tolist())

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

    def _episode_curve_image(self, policy):
        """MSE-vs-episode-timestep line plot as a wandb.Image (None if unavailable)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import wandb
        except Exception:
            return None
        res = self.metric.compute_mse(policy, self.ep_obs_dict, self.ep_gt)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.ep_timesteps, np.clip(res['scalar'], 1e-12, None),
                marker='.', linewidth=1.0, label='full chunk')
        ax.plot(self.ep_timesteps, np.clip(res['first'], 1e-12, None),
                marker='.', linewidth=1.0, label='first executed')
        for t in self.timesteps:  # mark the quarter points
            ax.axvline(t, color='k', linestyle=':', alpha=0.3)
        ax.set_yscale('log')
        ax.set_xlabel('episode rollout timestep')
        ax.set_ylabel('sampled-action MSE (log)')
        ax.set_title(f'{self.metric_prefix} vs episode timestep (episode {self.episode_index})')
        ax.legend()
        fig.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return img

    def compute(self, policy):
        self._log_summary_once()
        res = self.metric.compute_mse(policy, self.obs_dict, self.gt)
        log = dict()
        for i, lbl in enumerate(self.labels):
            log[f'{self.metric_prefix}/{lbl}'] = float(res['scalar'][i])
            log[f'{self.metric_prefix}_first/{lbl}'] = float(res['first'][i])

        if self.episode_curve and self.ep_obs_dict is not None:
            if (self._curve_call_count % self.episode_curve_every) == 0:
                img = self._episode_curve_image(policy)
                if img is not None:
                    log[f'{self.metric_prefix}_episode_curve'] = img
            self._curve_call_count += 1
        return log
