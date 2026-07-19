"""Symmetric history construction via a fake wrapper (no robosuite/render).

The fake obs = state-id (+1000 if the frame was 'perturbed'), so we can verify
exactly which frames each history mode uses and whether the applier touched them.
"""
import numpy as np

from diffusion_policy.experiments.spatial_attention_prelim_perturb.obs_builder import obs_builder

STATES = np.arange(10, dtype=float)  # state id == index


class _Sim:
    def __init__(self):
        self.perturbed = False

    def forward(self):
        pass


class _Inner:
    def __init__(self):
        self.sim = _Sim()


class _Env:
    def __init__(self):
        self.env = _Inner()
        self.cur = None

    def reset_to(self, d):
        self.env.sim.perturbed = False
        self.cur = d['states']


class _Wrapper:
    def __init__(self):
        self.env = _Env()

    def get_observation(self):
        v = float(self.env.cur) + (1000.0 if self.env.env.sim.perturbed else 0.0)
        return np.array([v], dtype=np.float32)


def _perturb_applier(sim, rs_env):
    sim.perturbed = True


def _obs(mode, t, applier):
    w = _Wrapper()
    d = obs_builder.build_input(w, 'lowdim', 2, mode, STATES, t, applier)
    return d['obs'].numpy()[0, :, 0]  # (To,)


def test_tile_perturbed_symmetry():
    assert np.allclose(_obs('tile_perturbed', 5, None), [5, 5])          # nominal
    assert np.allclose(_obs('tile_perturbed', 5, _perturb_applier), [1005, 1005])


def test_current_frame_only_history_is_nominal():
    # history frame nominal on BOTH sides; only the last frame differs
    assert np.allclose(_obs('current_frame_only', 5, None), [4, 5])
    assert np.allclose(_obs('current_frame_only', 5, _perturb_applier), [4, 1005])


def test_consistent_perturbed_applies_all_frames():
    assert np.allclose(_obs('consistent_perturbed', 5, None), [4, 5])
    assert np.allclose(_obs('consistent_perturbed', 5, _perturb_applier), [1004, 1005])


def test_t0_front_clamped():
    assert np.allclose(_obs('consistent_perturbed', 0, _perturb_applier), [1000, 1000])
    assert np.allclose(_obs('tile_perturbed', 0, None), [0, 0])


def test_same_shape_both_sides_all_modes():
    for mode in obs_builder.HISTORY_MODES:
        nom = _obs(mode, 5, None)
        pert = _obs(mode, 5, _perturb_applier)
        assert nom.shape == pert.shape == (2,)
