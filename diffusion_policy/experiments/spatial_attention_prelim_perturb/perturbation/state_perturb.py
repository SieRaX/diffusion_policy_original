"""Sample and apply a grasp-aware SE(3) perturbation to the MuJoCo state.

Sampling and applying are separated so the SAME realization can be applied to
several frames (consistent-history mode) while the grasp rule is re-evaluated
per frame.

- Non-grasped body: independent SE(3) noise (object sigmas).
- Grasped body: a single rigid ΔT (drawn with the EEF sigmas) is applied
  left-multiplied `T' = ΔT·T` to EVERY grasped body (shared ΔT), so their mutual
  relative pose is preserved; no independent noise on a grasped body.

`apply` writes each body's free-joint pose, zeroes that joint's velocity, and
`sim.forward()`s (no physics steps by default — settling would change the probed
state). perturb_targets gates what moves: default `['objects']`. `'eef'` is not
supported (needs a robosuite IK route to write an EEF delta into arm qpos) — it
raises, and the caller records the fallback. Under objects-only the EEF observation
is held fixed, so a grasped-body perturbation does not reflect a full physical grasp
motion (documented limitation).
"""
import numpy as np

from diffusion_policy.experiments.spatial_attention_prelim_perturb.perturbation import se3


class PerturbationRealization:
    """A fixed noise draw: a shared grasp ΔT plus a per-body independent delta.
    Applying re-uses these deltas; the grasp flag decides which to use per body."""
    def __init__(self, grasp_dpos, grasp_dquat, body_deltas):
        self.grasp_dpos = grasp_dpos          # (3,)
        self.grasp_dquat = grasp_dquat        # (4,) wxyz
        self.body_deltas = body_deltas        # {name: (dpos(3,), dquat(4,))}


def sample_realization(bodies, rng,
                       sigma_pos_eef, sigma_rot_eef,
                       sigma_pos_obj, sigma_rot_obj,
                       per_body_sigma=None):
    per_body_sigma = per_body_sigma or {}
    grasp_dpos, grasp_dquat = se3.sample_delta_transform(rng, sigma_pos_eef, sigma_rot_eef)
    body_deltas = {}
    for body in bodies:
        sp, sr = per_body_sigma.get(body.name, (sigma_pos_obj, sigma_rot_obj))
        dpos = se3.sample_position_noise(rng, sp)
        dquat = se3.sample_small_rotation_quat(rng, sr)
        body_deltas[body.name] = (dpos, dquat)
    return PerturbationRealization(grasp_dpos, grasp_dquat, body_deltas)


def _zero_joint_qvel(sim, jname):
    start, end = sim.model.get_joint_qvel_addr(jname)
    sim.data.qvel[start:end] = 0.0


def apply_realization(sim, bodies, grasp_flags, realization,
                      perturb_targets=('objects',), settle_steps=0):
    """Apply `realization` to `sim` in place (caller has reset_to a state)."""
    perturb_targets = list(perturb_targets)
    if 'eef' in perturb_targets:
        raise NotImplementedError(
            "perturb_targets includes 'eef': writing a small EEF SE(3) delta into the "
            "arm qpos needs a robosuite IK route (TODO). Use perturb_targets=['objects'].")
    if 'objects' not in perturb_targets:
        return

    for body, grasped in zip(bodies, grasp_flags):
        pose = np.array(sim.data.get_joint_qpos(body.joint), dtype=np.float64).copy()
        pos, quat = pose[:3], pose[3:]
        if grasped:
            new_pos, new_quat = se3.apply_delta_transform(
                realization.grasp_dpos, realization.grasp_dquat, pos, quat)
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
