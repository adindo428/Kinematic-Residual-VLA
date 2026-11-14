"""Inspect negotiated RealSense USB links without opening image streams."""
import argparse
import json
import sys

RIG_SERIALS={'front':'234222300865','side':'218722271409','wrist':'218622278247'}


def require_usb3(devices,required):
    found={d['serial_number']:d for d in devices};result={}
    for role,serial in required.items():
        if serial not in found:raise ValueError(f'{role} camera {serial} is not connected')
        info=found[serial];version=info.get('usb_type_descriptor')
        try:major=int(str(version).split('.')[0])
        except (TypeError,ValueError):raise ValueError(f'Cannot verify {role} camera USB link ({version!r})')
        if major<3:
            raise ValueError(f'{role} camera {serial} negotiated USB {version}; USB 3 is required for this RGB-D evaluation. '
                             'Use a USB 3 data cable and USB 3 port directly on the workstation. Keep its mount fixed.')
        result[role]=info
    return result


def resolve_front_color_format(device,requested='auto'):
    version=device.get('usb_type_descriptor')
    try:major=int(str(version).split('.')[0])
    except (TypeError,ValueError):raise ValueError('Cannot verify front camera USB link')
    if major<2:raise ValueError('Front camera requires at least a USB 2 link')
    if requested not in ('auto','bgr8','yuyv'):raise ValueError('Unsupported front color format')
    resolved=('yuyv' if major==2 else 'bgr8') if requested=='auto' else requested
    if major==2 and resolved!='yuyv':raise ValueError('Front USB 2 capture requires --front-color-format auto or yuyv at 30 FPS')
    return resolved


def validate_transport(devices,required,*,fps=30,color_format='auto'):
    if fps!=30:raise ValueError('VLA evaluation requires 30 FPS capture')
    found={d['serial_number']:d for d in devices}
    for role,serial in required.items():
        if serial not in found:raise ValueError(f'{role} camera {serial} is not connected')
    require_usb3(devices,{role:serial for role,serial in required.items() if role!='front'})
    if 'front' in required:resolve_front_color_format(found[required['front']],color_format)
    return {role:found[serial] for role,serial in required.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-usb3',action='store_true',help='Return a failure exit code if a rig camera is below USB 3')
    args=parser.parse_args()
    from .camera import enumerate_cameras
    devices=enumerate_cameras();by_serial={d['serial_number']:d for d in devices}
    print(json.dumps({role:by_serial.get(serial,{'serial_number':serial,'state':'missing'}) for role,serial in RIG_SERIALS.items()},indent=2))
    if args.require_usb3:
        try:require_usb3(devices,RIG_SERIALS)
        except ValueError as exc:print(str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
