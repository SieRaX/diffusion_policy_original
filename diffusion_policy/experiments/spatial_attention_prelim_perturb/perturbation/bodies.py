"""Resolve the perturbable bodies for a task from the robosuite env handles.

We read authoritative names off the object handles (`obj.name`, `obj.joints[0]`,
`obj.root_body`, `obj.contact_geoms`) rather than hardcoding, and assert the free
joint exists in the model. A per-task-base default lists which object handles to
perturb; config may override with an explicit list of object names.
Fixed fixtures (e.g. Square's peg, ToolHang's stand) are never perturbed.
"""

# object NAMES to perturb, keyed by robosuite task base
DEFAULT_TASK_OBJECTS = {
    'lift': ['cube'],
    'can': ['Can'],
    'square': ['SquareNut'],
    'tool_hang': ['tool', 'frame'],   # 'frame' is not graspable; it gets independent noise
}


class PerturbBody:
    def __init__(self, name, joint, root_body, grasp_handle):
        self.name = name
        self.joint = joint              # free-joint name, e.g. 'cube_joint0'
        self.root_body = root_body
        self.grasp_handle = grasp_handle  # robosuite MujocoObject (for _check_grasp), or None

    def __repr__(self):
        return f"PerturbBody(name={self.name!r}, joint={self.joint!r})"


def task_base_from_name(cfg_task):
    """Robosuite task base ('lift'/'can'/'square'/'tool_hang') from the checkpoint cfg.task."""
    tn = cfg_task.get('task_name', None)
    if tn is not None:
        return str(tn)
    name = str(cfg_task['name'])
    for suf in ('_lowdim_abs', '_image_abs', '_lowdim', '_image', '_abs'):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def all_object_handles(rs_env):
    """Gather {name: handle} for every free-body object handle exposed by the env."""
    handles = {}
    for attr in ('cube', 'tool', 'frame', 'stand'):
        obj = getattr(rs_env, attr, None)
        if obj is not None and hasattr(obj, 'joints'):
            handles[obj.name] = obj
    for attr in ('objects', 'nuts'):
        lst = getattr(rs_env, attr, None)
        if lst is not None:
            for obj in lst:
                if hasattr(obj, 'joints') and getattr(obj, 'joints', None):
                    handles[obj.name] = obj
    return handles


def resolve_perturb_bodies(rs_env, sim, task_base, object_names_override=None):
    """Return a list of PerturbBody for the task. `object_names_override` (list of
    robosuite object names) replaces the per-task default when provided."""
    handles = all_object_handles(rs_env)
    if object_names_override:
        names = list(object_names_override)
    elif task_base in DEFAULT_TASK_OBJECTS:
        names = DEFAULT_TASK_OBJECTS[task_base]
    else:
        raise KeyError(
            f"No object_bodies lookup for task base '{task_base}'. Provide "
            f"cfg.object_bodies override. Known bases: {sorted(DEFAULT_TASK_OBJECTS)}")

    joint_names = set(sim.model.joint_names)
    bodies = []
    for name in names:
        handle = handles.get(name, None)
        if handle is None:
            raise KeyError(
                f"Object handle '{name}' not found on env. Available: {sorted(handles)}")
        joint = handle.joints[0]
        assert joint in joint_names, \
            f"Free joint '{joint}' for object '{name}' not in model joints: {sorted(joint_names)}"
        root_body = getattr(handle, 'root_body', None)
        bodies.append(PerturbBody(name=name, joint=joint, root_body=root_body, grasp_handle=handle))
    return bodies
