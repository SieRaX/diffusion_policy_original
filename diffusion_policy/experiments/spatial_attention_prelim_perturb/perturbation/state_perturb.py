"""Sample and apply a grasp-aware SE(3) perturbation to the MuJoCo state.

Sampling and applying are separated so the SAME realization can be applied to
several frames (consistent-history mode) while the grasp rule is re-evaluated
per frame.

Two perturbation sources, selected by ``perturb_targets``:
- ``objects`` (default): perturb the free-body objects directly (their free-joint
  qpos IS their pose). Non-grasped body -> independent SE(3) noise. Grasped body ->
  a single SAMPLED rigid ΔT (EEF sigmas) applied to every grasped body (shared),
  preserving their mutual relative pose. The EEF observation is held fixed here.
- ``eef``: perturb the arm joints directly (Gaussian noise on the arm qpos — the
  clean, IK-free way to move the end-effector; note this is JOINT-space noise, so
  the induced EEF Cartesian displacement depends on the arm configuration). The
  EEF's INDUCED rigid transform ΔT is read from forward kinematics before/after,
  and every grasped body is carried by that ΔT so the grasp stays rigid.

``eef`` and ``objects`` may be combined: the arm is nudged, grasped bodies follow
the induced ΔT, and non-grasped bodies additionally get independent noise.

``apply`` writes poses, zeroes the affected joint velocities, and ``sim.forward()``s
(no physics steps by default — settling would change the probed state).
"""
import numpy as np

from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation import se3


class PerturbationRealization:
    """A fixed noise draw: a shared grasp ΔT (objects mode), per-body independent
    deltas, and arm-qpos noise (eef mode)."""
    def __init__(self, grasp_dpos, grasp_dquat, body_deltas, arm_qpos_noise=None):
        self.grasp_dpos = grasp_dpos          # (3,)
        self.grasp_dquat = grasp_dquat        # (4,) wxyz
        self.body_deltas = body_deltas        # {name: (dpos(3,), dquat(4,))}
        self.arm_qpos_noise = arm_qpos_noise  # (n_arm,) or None


def sample_realization(bodies, rng,
                       sigma_pos_eef, sigma_rot_eef,
                       sigma_pos_obj, sigma_rot_obj,
                       per_body_sigma=None,
                       sigma_qpos=0.0, n_arm_joints=None):
    per_body_sigma = per_body_sigma or {}
    # sample objects-mode noise first so existing objects-only runs are unchanged
    grasp_dpos, grasp_dquat = se3.sample_delta_transform(rng, sigma_pos_eef, sigma_rot_eef)
    body_deltas = {}
    for body in bodies:
        sp, sr = per_body_sigma.get(body.name, (sigma_pos_obj, sigma_rot_obj))
        dpos = se3.sample_position_noise(rng, sp)
        dquat = se3.sample_small_rotation_quat(rng, sr)
        body_deltas[body.name] = (dpos, dquat)
    arm_qpos_noise = None
    if n_arm_joints:
        arm_qpos_noise = rng.normal(0.0, sigma_qpos, size=int(n_arm_joints))
    return PerturbationRealization(grasp_dpos, grasp_dquat, body_deltas, arm_qpos_noise)


def _zero_joint_qvel(sim, jname):
    start, end = sim.model.get_joint_qvel_addr(jname)
    sim.data.qvel[start:end] = 0.0


def eef_pose(rs_env, sim):
    """End-effector pose (grip-site position + orientation) as (pos, quat_wxyz)."""
    site = rs_env.robots[0].gripper.important_sites['grip_site']  # 'gripper0_grip_site'
    sid = sim.model.site_name2id(site)
    pos = np.array(sim.data.site_xpos[sid], dtype=np.float64)
    R = np.array(sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    return pos, se3.mat_to_quat(R)


def _apply_arm_noise(sim, rs_env, arm_qpos_noise):
    """Add joint noise to the arm qpos, zero arm qvel, forward. Returns the induced
    EEF rigid transform ΔT = (dpos, dquat)."""
    p0, q0 = eef_pose(rs_env, sim)
    robot = rs_env.robots[0]
    idx = robot._ref_joint_pos_indexes
    sim.data.qpos[idx] += np.asarray(arm_qpos_noise, dtype=np.float64)
    vidx = robot._ref_joint_vel_indexes
    sim.data.qvel[vidx] = 0.0
    sim.forward()
    p1, q1 = eef_pose(rs_env, sim)
    return se3.compose_delta(p0, q0, p1, q1)


def apply_realization(sim, bodies, grasp_flags, realization, rs_env=None,
                      perturb_targets=('objects',), settle_steps=0):
    """Apply `realization` to `sim` in place (caller has reset_to a state)."""
    perturb_targets = list(perturb_targets)
    do_eef = 'eef' in perturb_targets
    do_objects = 'objects' in perturb_targets
    if not (do_eef or do_objects):
        return

    if do_eef:
        assert rs_env is not None and realization.arm_qpos_noise is not None, \
            "eef perturbation needs rs_env and a sampled arm_qpos_noise"
        # grasped bodies follow the EEF's INDUCED transform after the arm nudge
        grasp_dpos, grasp_dquat = _apply_arm_noise(sim, rs_env, realization.arm_qpos_noise)
    else:
        grasp_dpos, grasp_dquat = realization.grasp_dpos, realization.grasp_dquat

    for body, grasped in zip(bodies, grasp_flags):
        if not grasped and not do_objects:
            continue  # non-grasped body, objects not targeted -> leave it
        pose = np.array(sim.data.get_joint_qpos(body.joint), dtype=np.float64).copy()
        pos, quat = pose[:3], pose[3:]
        if grasped:
            new_pos, new_quat = se3.apply_delta_transform(grasp_dpos, grasp_dquat, pos, quat)
        else:
            dpos, dquat = realization.body_deltas[body.name]
            new_pos = pos + np.asarray(dpos, dtype=np.float64)
            new_quat = se3.quat_normalize(se3.quat_mul(se3.quat_normalize(dquat),
                                                       se3.quat_normalize(quat)))
        sim.data.set_joint_qpos(body.joint, np.concatenate([new_pos, new_quat]))
        _zero_joint_qvel(sim, body.joint)

    for _ in range(int(settle_steps)):
        sim.step()
    sim.forward()
