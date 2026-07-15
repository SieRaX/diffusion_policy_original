from diffusion_policy.experiments.spatial_attention_exp1.mse_metric.state_provider import (
    resolve_quarter_timesteps,
)

FRACS = [0.0, 0.25, 0.5, 0.75]


def test_regular_episode():
    pairs = resolve_quarter_timesteps(100, FRACS)
    labels = [p[0] for p in pairs]
    ts = [p[1] for p in pairs]
    assert labels == ['frac00', 'frac25', 'frac50', 'frac75']
    assert ts == [0, 25, 50, 75]


def test_one_per_fraction_even_when_collapsing():
    # short episode: fractions may collapse onto the same timestep, but we still
    # emit ONE (label, timestep) per requested fraction (the wandb hook logs per frac)
    pairs = resolve_quarter_timesteps(2, FRACS)
    assert [p[1] for p in pairs] == [0, 0, 1, 1]
    assert [p[0] for p in pairs] == ['frac00', 'frac25', 'frac50', 'frac75']


def test_length_one_episode():
    # last_timestep == 0 -> everything collapses to 0
    pairs = resolve_quarter_timesteps(0, FRACS)
    assert [p[1] for p in pairs] == [0, 0, 0, 0]


def test_clamping():
    pairs = resolve_quarter_timesteps(10, [0.0, 0.99, 1.0, 1.5])
    ts = [p[1] for p in pairs]
    assert max(ts) <= 10 and min(ts) >= 0
