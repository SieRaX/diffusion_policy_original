"""Common-random-number (CRN) management for the per-state MSE experiment.

Two fixed random sets are drawn ONCE from fixed seeds and reused for every state,
every rollout event, and every checkpoint (within and across variants sharing the
same (horizon, action_dim)):

  * ``eps0``   : the K_s initial-noise chunks for the ODE sampler, shape
                 ``(K_s, H, D)``. Fed into the policy via its ``_init_noise``
                 attribute so ``MSE(o; theta) = mean_j || f_theta(eps0_j, o) - a_GT ||^2``
                 uses the SAME noise everywhere.
  * ``taus`` / ``eps_fm`` : the fixed ``(tau_j, eps_j)`` set (default 32) for the
                 secondary FM-loss. ``taus`` are stratified on a uniform grid over
                 [0, 1] — one sample per stratum ``[j/N, (j+1)/N)``.

Tensors are generated on CPU with explicit ``torch.Generator`` seeds so they are
bit-identical regardless of GPU/device; move to the compute device at use time.
Because the draw depends only on ``(seed, shape)``, two CRNManagers with matching
seeds and matching ``(H, D)`` produce identical sets — this is what makes the CRN
reusable across variants.
"""
import torch


class CRNManager:
    def __init__(self,
            k_s: int,
            horizon: int,
            action_dim: int,
            num_fm: int = 32,
            eps0_seed: int = 0,
            fm_seed: int = 1):
        self.k_s = int(k_s)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.num_fm = int(num_fm)
        self.eps0_seed = int(eps0_seed)
        self.fm_seed = int(fm_seed)

        # --- initial-noise CRN set {eps0_j}, shape (K_s, H, D) ---
        g0 = torch.Generator()
        g0.manual_seed(self.eps0_seed)
        self.eps0 = torch.randn(
            (self.k_s, self.horizon, self.action_dim), generator=g0)

        # --- FM-loss CRN set: stratified taus + eps_fm, drawn from one generator ---
        gf = torch.Generator()
        gf.manual_seed(self.fm_seed)
        u = torch.rand((self.num_fm,), generator=gf)                 # one per stratum
        self.taus = (torch.arange(self.num_fm) + u) / self.num_fm     # in [0, 1)
        self.eps_fm = torch.randn(
            (self.num_fm, self.horizon, self.action_dim), generator=gf)

    def metadata(self) -> dict:
        return {
            'crn_k_s': self.k_s,
            'crn_num_fm': self.num_fm,
            'crn_horizon': self.horizon,
            'crn_action_dim': self.action_dim,
            'crn_eps0_seed': self.eps0_seed,
            'crn_fm_seed': self.fm_seed,
        }
