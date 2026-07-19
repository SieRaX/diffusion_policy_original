"""SE(3) perturbation math (numpy). Quaternions are MuJoCo-native (w, x, y, z).

Orientation noise NEVER adds Gaussian noise to quaternion components directly:
we sample an axis uniformly on the sphere and an angle ~ N(0, sigma_rot^2), build
the corresponding unit quaternion, and left-multiply it onto the existing
orientation (renormalizing). Position noise is Gaussian in meters.
"""
import numpy as np


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(q1, q2):
    """Hamilton product of two (w,x,y,z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_mat(q):
    """(w,x,y,z) unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_angle(q):
    """Rotation angle (radians) of a (w,x,y,z) quaternion, in [0, pi]."""
    w = abs(float(quat_normalize(q)[0]))
    return 2.0 * np.arccos(np.clip(w, -1.0, 1.0))


def random_unit_vector(rng):
    """Axis sampled uniformly on the unit sphere."""
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    while n < 1e-9:  # astronomically unlikely; guard anyway
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
    return v / n


def sample_small_rotation_quat(rng, sigma_rot):
    """Axis uniform on the sphere, angle ~ N(0, sigma_rot^2) -> unit quat (w,x,y,z)."""
    axis = random_unit_vector(rng)
    angle = float(rng.normal(0.0, sigma_rot))
    half = 0.5 * angle
    q = np.concatenate([[np.cos(half)], np.sin(half) * axis])
    return quat_normalize(q)


def sample_position_noise(rng, sigma_pos):
    return rng.normal(0.0, sigma_pos, size=3)


def perturb_pose_independent(rng, pos, quat, sigma_pos, sigma_rot):
    """Independent SE(3) noise on a single body pose (non-grasped case)."""
    dpos = sample_position_noise(rng, sigma_pos)
    dquat = sample_small_rotation_quat(rng, sigma_rot)
    new_pos = np.asarray(pos, dtype=np.float64) + dpos
    new_quat = quat_normalize(quat_mul(dquat, quat_normalize(quat)))
    return new_pos, new_quat


def sample_delta_transform(rng, sigma_pos, sigma_rot):
    """A rigid SE(3) increment ΔT = (dpos, dquat), drawn with the EEF sigmas.
    Used for grasp-aware coupling (applied to every grasped body)."""
    dpos = sample_position_noise(rng, sigma_pos)
    dquat = sample_small_rotation_quat(rng, sigma_rot)
    return dpos, dquat


def apply_delta_transform(dpos, dquat, pos, quat):
    """Left-multiply a world-frame rigid transform: T' = ΔT · T.
    Preserves the relative pose between any two bodies transformed by the same ΔT
    (grasp coupling). p' = R(dquat)·p + dpos ; R' = R(dquat)·R."""
    R = quat_to_mat(dquat)
    new_pos = R @ np.asarray(pos, dtype=np.float64) + np.asarray(dpos, dtype=np.float64)
    new_quat = quat_normalize(quat_mul(quat_normalize(dquat), quat_normalize(quat)))
    return new_pos, new_quat
