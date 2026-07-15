import torch

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.crn import CRNManager


def test_crn_deterministic_across_instances():
    a = CRNManager(k_s=16, horizon=8, action_dim=10, num_fm=32, eps0_seed=0, fm_seed=1)
    b = CRNManager(k_s=16, horizon=8, action_dim=10, num_fm=32, eps0_seed=0, fm_seed=1)
    # identical {eps0_j}, {(tau_j, eps_j)} across independently constructed managers
    assert torch.equal(a.eps0, b.eps0)
    assert torch.equal(a.taus, b.taus)
    assert torch.equal(a.eps_fm, b.eps_fm)


def test_crn_reuse_across_variants_same_shape():
    # "across variants" == same (H, D) and seeds -> identical CRN sets
    lowdim = CRNManager(k_s=16, horizon=16, action_dim=10, eps0_seed=0, fm_seed=1)
    image = CRNManager(k_s=16, horizon=16, action_dim=10, eps0_seed=0, fm_seed=1)
    assert torch.equal(lowdim.eps0, image.eps0)


def test_crn_seed_changes_noise():
    a = CRNManager(k_s=4, horizon=4, action_dim=7, eps0_seed=0)
    b = CRNManager(k_s=4, horizon=4, action_dim=7, eps0_seed=123)
    assert not torch.equal(a.eps0, b.eps0)


def test_crn_shapes():
    c = CRNManager(k_s=16, horizon=8, action_dim=10, num_fm=32)
    assert tuple(c.eps0.shape) == (16, 8, 10)
    assert tuple(c.eps_fm.shape) == (32, 8, 10)
    assert tuple(c.taus.shape) == (32,)


def test_taus_stratified_uniform_grid():
    num_fm = 32
    c = CRNManager(k_s=2, horizon=2, action_dim=2, num_fm=num_fm)
    taus = c.taus.numpy()
    assert taus.min() >= 0.0 and taus.max() < 1.0
    # exactly one tau per stratum [j/N, (j+1)/N)
    strata = (taus * num_fm).astype(int)
    assert sorted(strata.tolist()) == list(range(num_fm))
