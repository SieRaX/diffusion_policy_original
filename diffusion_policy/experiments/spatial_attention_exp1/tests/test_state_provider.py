import numpy as np

from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.state_provider import (
    build_subsample,
)


def test_subsample_deterministic():
    a = build_subsample(1000, 100, seed=0)
    b = build_subsample(1000, 100, seed=0)
    assert np.array_equal(a, b)


def test_subsample_seed_changes():
    a = build_subsample(1000, 100, seed=0)
    b = build_subsample(1000, 100, seed=1)
    assert not np.array_equal(a, b)


def test_subsample_includes_must_include():
    must = [7, 42, 999]
    idx = build_subsample(1000, 50, seed=3, must_include=must)
    for m in must:
        assert m in idx
    # sorted + unique
    assert np.array_equal(idx, np.unique(idx))


def test_subsample_size_ge_n_returns_all():
    idx = build_subsample(20, 100, seed=0)
    assert np.array_equal(idx, np.arange(20))


def test_subsample_size_ge_n_still_includes_must():
    idx = build_subsample(20, 100, seed=0, must_include=[5])
    assert 5 in idx
