import pytest
from real_robot.camera_transport import require_usb3


def test_usb2_front_is_rejected_before_streams_start():
    devices=[dict(serial_number='front',usb_type_descriptor='2.1'),dict(serial_number='side',usb_type_descriptor='3.2')]
    with pytest.raises(ValueError,match='front camera front negotiated USB 2.1'):
        require_usb3(devices,{'front':'front','side':'side'})


def test_superspeed_cameras_are_accepted_and_role_mapping_is_preserved():
    devices=[dict(serial_number='a',usb_type_descriptor='3.2',physical_port='/usb2/2-1'),
             dict(serial_number='b',usb_type_descriptor='3.0',physical_port='/usb2/2-4')]
    result=require_usb3(devices,{'front':'b','side':'a'})
    assert result['front']['physical_port']=='/usb2/2-4'
    assert result['side']['serial_number']=='a'


def test_missing_or_unknown_link_is_not_assumed_usb3():
    with pytest.raises(ValueError,match='not connected'):require_usb3([],{'front':'a'})
    with pytest.raises(ValueError,match='Cannot verify'):require_usb3([{'serial_number':'a'}],{'front':'a'})


def test_usb2_front_automatically_uses_packed_30fps_and_requires_usb3_wrist():
    from real_robot.camera_transport import validate_transport,resolve_front_color_format
    devices=[dict(serial_number='front',usb_type_descriptor='2.1'),dict(serial_number='wrist',usb_type_descriptor='3.2')]
    roles={'front':'front','wrist':'wrist'}
    result=validate_transport(devices,roles)
    assert resolve_front_color_format(result['front'])=='yuyv'
    with pytest.raises(ValueError,match='30 FPS'):validate_transport(devices,roles,fps=15)
    with pytest.raises(ValueError,match='yuyv'):validate_transport(devices,roles,color_format='bgr8')
    assert validate_transport(devices,roles,color_format='yuyv')['front']['usb_type_descriptor']=='2.1'
    devices[1]['usb_type_descriptor']='2.1'
    with pytest.raises(ValueError,match='wrist camera'):validate_transport(devices,roles)


def test_auto_format_uses_bgr_on_usb3_and_rejects_unknown_front_link():
    from real_robot.camera_transport import validate_transport,resolve_front_color_format
    devices=[dict(serial_number='front',usb_type_descriptor='3.2')]
    assert resolve_front_color_format(validate_transport(devices,{'front':'front'})['front'])=='bgr8'
    assert resolve_front_color_format(devices[0],'yuyv')=='yuyv'
    devices[0].pop('usb_type_descriptor')
    with pytest.raises(ValueError,match='Cannot verify'):validate_transport(devices,{'front':'front'})
