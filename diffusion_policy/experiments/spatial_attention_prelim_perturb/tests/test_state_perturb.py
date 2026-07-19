"""Grasp-aware coupling via a fake MuJoCo sim (no robosuite)."""
import numpy as np

from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation import se3
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.bodies import PerturbBody
from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation.state_perturb import (
    sample_realization, apply_realization,
)


class _FakeModel:
    def __init__(self, qvel_addr):
        self._qvel_addr = qvel_addr

    def get_joint_qvel_addr(self, name):
        return self._qvel_addr[name]


class _FakeData:
    def __init__(self, poses):
        self.qpos_by_joint = {k: np.asarray(v, dtype=np.float64).copy() for k, v in poses.items()}
        self.qvel = np.ones(64)

    def get_joint_qpos(self, name):
        return self.qpos_by_joint[name].copy()

    def set_joint_qpos(self, name, val):
        self.qpos_by_joint[name] = np.asarray(val, dtype=np.float64).copy()


class _FakeSim:
    def __init__(self, poses, qvel_addr):
        self.data = _FakeData(poses)
        self.model = _FakeModel(qvel_addr)

    def forward(self):
        pass

    def step(self):
        pass


def _body(name):
    return PerturbBody(name=name, joint=name, root_body=name + '_main', grasp_handle=None)


def _pose(pos, quat):
    return np.concatenate([pos, se3.quat_normalize(quat)])


def test_two_grasped_bodies_follow_shared_transform():
    poses = {
        'a': _pose([0.4, 0.0, 0.85], [1, 0.1, 0, 0]),
        'b': _pose([0.45, 0.05, 0.9], [1, 0, 0.2, 0]),
    }
    qvel_addr = {'a': (0, 6), 'b': (6, 12)}
    sim = _FakeSim(poses, qvel_addr)
    bodies = [_body('a'), _body('b')]
    rng = np.random.default_rng(0)
    real = sample_realization(bodies, rng, 0.01, 0.05, 0.005, 0.02)

    # relative position (a-frame) before
    pa0 = poses['a'][:3]; qa0 = poses['a'][3:]; pb0 = poses['b'][:3]
    rel_before = se3.quat_to_mat(qa0).T @ (pb0 - pa0)

    apply_realization(sim, bodies, [True, True], real)  # both grasped -> shared ΔT

    pa1 = sim.data.get_joint_qpos('a'); pb1 = sim.data.get_joint_qpos('b')
    rel_after = se3.quat_to_mat(pa1[3:]).T @ (pb1[:3] - pa1[:3])
    assert np.allclose(rel_before, rel_after, atol=1e-9)
    # grasped bodies actually moved
    assert not np.allclose(pa1[:3], pa0)
    # qvel zeroed
    assert np.allclose(sim.data.qvel[0:6], 0.0) and np.allclose(sim.data.qvel[6:12], 0.0)


def test_nongrasped_body_perturbed_independently():
    poses = {
        'grasped': _pose([0.4, 0.0, 0.85], [1, 0, 0, 0]),
        'free': _pose([0.6, 0.2, 0.85], [1, 0, 0, 0]),
    }
    qvel_addr = {'grasped': (0, 6), 'free': (6, 12)}
    sim = _FakeSim(poses, qvel_addr)
    bodies = [_body('grasped'), _body('free')]
    rng = np.random.default_rng(1)
    real = sample_realization(bodies, rng, 0.01, 0.05, 0.005, 0.02)

    apply_realization(sim, bodies, [True, False], real)
    free_after = sim.data.get_joint_qpos('free')
    # the free body moved by its OWN independent delta (not the grasp ΔT)
    expected_pos = poses['free'][:3] + real.body_deltas['free'][0]
    assert np.allclose(free_after[:3], expected_pos, atol=1e-9)


def test_eef_target_raises():
    sim = _FakeSim({'a': _pose([0, 0, 0], [1, 0, 0, 0])}, {'a': (0, 6)})
    import pytest
    with pytest.raises(NotImplementedError):
        apply_realization(sim, [_body('a')], [False],
                          sample_realization([_body('a')], np.random.default_rng(0),
                                             0.01, 0.05, 0.005, 0.02),
                          perturb_targets=('eef', 'objects'))
