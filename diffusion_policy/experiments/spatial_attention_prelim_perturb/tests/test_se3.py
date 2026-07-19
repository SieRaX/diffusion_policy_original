import numpy as np

from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation import se3


def test_small_rotation_quat_is_unit_and_small():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = se3.sample_small_rotation_quat(rng, sigma_rot=0.02)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9      # unit norm
        assert np.all(np.isfinite(q))
        assert se3.quat_angle(q) < 0.3                   # small (sigma=0.02 rad)


def test_quat_mul_identity():
    ident = np.array([1.0, 0, 0, 0])
    rng = np.random.default_rng(1)
    q = se3.quat_normalize(rng.normal(size=4))
    assert np.allclose(se3.quat_mul(ident, q), q)
    assert np.allclose(se3.quat_mul(q, ident), q)


def test_perturb_pose_independent_keeps_unit_quat():
    rng = np.random.default_rng(2)
    pos = np.array([0.5, 0.1, 0.9])
    quat = se3.quat_normalize(np.array([1.0, 0.2, -0.1, 0.05]))
    p2, q2 = se3.perturb_pose_independent(rng, pos, quat, 0.005, 0.02)
    assert abs(np.linalg.norm(q2) - 1.0) < 1e-9
    assert np.all(np.isfinite(p2)) and np.all(np.isfinite(q2))


def test_delta_transform_preserves_relative_pose():
    """Two bodies transformed by the SAME ΔT keep their relative pose (grasp coupling)."""
    rng = np.random.default_rng(3)
    p1 = np.array([0.4, 0.0, 0.85]); q1 = se3.quat_normalize(np.array([1.0, 0.1, 0.0, 0.0]))
    p2 = np.array([0.45, 0.05, 0.9]); q2 = se3.quat_normalize(np.array([1.0, 0.0, 0.2, 0.0]))
    dpos, dquat = se3.sample_delta_transform(rng, 0.01, 0.05)

    p1b, q1b = se3.apply_delta_transform(dpos, dquat, p1, q1)
    p2b, q2b = se3.apply_delta_transform(dpos, dquat, p2, q2)

    # relative position in body-1 frame: R1^T (p2 - p1)
    R1 = se3.quat_to_mat(q1); R1b = se3.quat_to_mat(q1b)
    rel_before = R1.T @ (p2 - p1)
    rel_after = R1b.T @ (p2b - p1b)
    assert np.allclose(rel_before, rel_after, atol=1e-9)
    # relative orientation q1^-1 q2 preserved
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    q1b_inv = np.array([q1b[0], -q1b[1], -q1b[2], -q1b[3]])
    rel_q = se3.quat_mul(q1_inv, q2)
    rel_qb = se3.quat_mul(q1b_inv, q2b)
    # quaternions equal up to sign
    assert np.allclose(rel_q, rel_qb, atol=1e-8) or np.allclose(rel_q, -rel_qb, atol=1e-8)
