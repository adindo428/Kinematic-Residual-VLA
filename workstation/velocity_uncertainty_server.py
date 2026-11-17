"""Matched CR simulator with measurement corruption; physical dynamics unchanged."""
import dom_repeat_extension_server as original

base_patch = original.full.patch_source


def patch_source(text):
    text = base_patch(text)
    needle = '    rcv_action = False'
    assert text.count(needle) == 1
    text = text.replace(needle, '    from velocity_uncertainty_sensor import VelocitySensor\n    _velocity_sensor = VelocitySensor()\n    _velocity_sensor.prepare_frame(env)\n'+needle)
    needle = '        cam_view = sim.get_camera_views('
    assert text.count(needle) == 1
    text = text.replace(needle, '        _velocity_sensor.prepare_frame(env)\n'+needle)
    needle = '        obs_socket.send_pyobj(\n            {\n                "dt_scale": ('
    assert text.count(needle) == 1
    text = text.replace(needle, '''        _measured_velocity = _velocity_sensor.measure(
            env, curr_state['object']['pos'], curr_state['object']['velocity'], len(sim_results['cam_views'])-1)
        curr_state['object'] = dict(curr_state['object'], velocity=torch.as_tensor(
            _measured_velocity, device=curr_state['object']['velocity'].device))
'''+needle)
    needle = '    _profile_ctrl.finish(_trace)'
    assert text.count(needle) == 1
    text = text.replace(needle, needle+'\n    _velocity_sensor.finish()')
    # Enable matching camera modalities for every arm, preserving the RGB view.
    needle = '"data_types": v["data_types"],'
    assert text.count(needle) == 1
    text = text.replace(needle, '"data_types": list(dict.fromkeys(v["data_types"] + ["distance_to_image_plane", "instance_segmentation_fast"])),')
    # Non-colorized instance IDs are required for the first-frame target mask.
    needle = '            configs.scene_cfg.get_camera_cfg('
    assert text.count(needle) == 1
    text = text.replace(needle, '            _uncertainty_camera_cfg(')
    return 'from velocity_uncertainty_server import camera_cfg as _uncertainty_camera_cfg\n'+text


def camera_cfg(*args, **kwargs):
    from configs.scene_cfg import get_camera_cfg
    cfg = get_camera_cfg(*args, **kwargs)
    cfg.colorize_instance_segmentation = False
    return cfg


if __name__ == '__main__':
    original.full.profile.server.prompt = original.prompt
    original.full.resident.patch_source = patch_source
    original.full.profile.server.main()
