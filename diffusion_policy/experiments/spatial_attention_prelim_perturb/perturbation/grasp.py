"""Per-body grasp detection: gripper closed beyond a threshold AND an active
MuJoCo contact between the gripper finger pads and the body.

Both are queried from the sim after reset_to + forward. `_check_grasp` (robosuite
ManipulationEnv) already requires contact on both finger pads; the finger-qpos gate
adds the "closed" condition. The qpos threshold is Panda/gripper-specific and
heuristic — it is config-overridable.
"""
import numpy as np


def gripper_closed(rs_env, sim, qpos_threshold):
    """True if the gripper's finger joints are closed beyond `qpos_threshold`
    (sum of |finger qpos| below threshold)."""
    try:
        idxs = rs_env.robots[0]._ref_gripper_joint_pos_indexes
    except Exception:
        return True  # no gripper joints exposed -> don't gate on closedness
    if idxs is None or len(idxs) == 0:
        return True
    fq = np.array([sim.data.qpos[i] for i in idxs], dtype=np.float64)
    return bool(np.sum(np.abs(fq)) < float(qpos_threshold))


def check_contact_grasp(rs_env, body):
    """Contact-based grasp via robosuite `_check_grasp` (both finger pads touch body)."""
    if body.grasp_handle is None:
        return False
    try:
        return bool(rs_env._check_grasp(
            gripper=rs_env.robots[0].gripper, object_geoms=body.grasp_handle))
    except Exception:
        return False


def is_grasped(rs_env, sim, body, qpos_threshold):
    return check_contact_grasp(rs_env, body) and gripper_closed(rs_env, sim, qpos_threshold)
