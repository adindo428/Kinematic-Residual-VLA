"""Change only the Franka finger collision materials, with native PhysX readback."""


def finger_shape_indices(names, shapes_per_body):
    if len(names) != len(shapes_per_body):
        raise ValueError('Body/shape mapping length mismatch')
    selected = []
    offset = 0
    fingers = []
    for name, count in zip(names, shapes_per_body):
        if name in ('panda_leftfinger', 'panda_rightfinger'):
            if count < 1:
                raise ValueError('Finger body has no collision shapes')
            fingers.append(name)
            selected.extend(range(offset, offset+count))
        offset += count
    if set(fingers) != {'panda_leftfinger', 'panda_rightfinger'}:
        raise ValueError('Both Franka finger bodies must be found')
    return selected


def update_materials(materials, indices, coefficient):
    import torch
    if coefficient is not None and (not 0 <= coefficient <= 10):
        raise ValueError('Contact coefficient must be in [0,10]')
    if not indices or min(indices) < 0 or max(indices) >= materials.shape[1]:
        raise ValueError('Invalid finger collision shape mapping')
    result = materials.clone()
    if coefficient is not None:
        result[:, indices, 0] = coefficient
        result[:, indices, 1] = coefficient
        if not torch.all(result[:, indices, :2] >= materials[:, indices, :2]):
            raise ValueError('Requested diagnostic value would reduce existing contact friction')
    return result


def apply(env, coefficient):
    import torch
    robot = env.unwrapped.scene['robot']
    target = env.unwrapped.scene['object']
    before = robot.root_physx_view.get_material_properties().clone()
    object_before = target.root_physx_view.get_material_properties().clone()
    counts = [robot._physics_sim_view.create_rigid_body_view(path).max_shapes
              for path in robot.root_physx_view.link_paths[0]]
    if sum(counts) != robot.root_physx_view.max_shapes:
        raise RuntimeError('Shape indexing does not match native articulation view')
    indices = finger_shape_indices(robot.body_names, counts)
    updated = update_materials(before, indices, coefficient)
    if coefficient is not None:
        robot.root_physx_view.set_material_properties(updated, torch.arange(env.unwrapped.num_envs, device='cpu'))
    after = robot.root_physx_view.get_material_properties().clone()
    torch.testing.assert_close(after, updated, rtol=0, atol=1e-6)
    object_after = target.root_physx_view.get_material_properties().clone()
    torch.testing.assert_close(object_after, object_before, rtol=0, atol=0)
    # Full-tensor equality to the intended update also verifies other bodies and restitution.
    return {'requested_finger_coefficient': coefficient, 'body_names': robot.body_names,
            'shapes_per_body': counts, 'finger_shape_indices': indices,
            'finger_materials_before': before[:, indices].tolist(),
            'finger_materials_after': after[:, indices].tolist(),
            'object_materials_unchanged': object_after.tolist(),
            'native_readback_verified': True,
            'nonfinger_materials_and_restitution_unchanged': True,
            'scope': 'Finger collision static/dynamic friction; joint friction and combine modes retained.'}
