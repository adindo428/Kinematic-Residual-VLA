"""Measure right ALOHA leader or follower gripper endpoints without commanding motors.

Run with the LeRobot Python environment. By default this only previews changes;
--save backs up and updates only the selected gripper endpoints. The default is
the leader. Use --arm follower --open-only to preserve the follower's closed
position and measure a wider opening. Stop teleoperation/collection first; rest
and support the follower with its gripper torque off before moving it manually.
Choose an opening short of the mechanical stop; never force the linkage.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time


DEFAULT_CALIBRATION = Path('/home/ailab/lerobot/.cache/calibration/aloha_default/right_leader.json')


def with_gripper_endpoints(calibration, closed, opened):
    """Preserve all existing calibration except the LINEAR gripper endpoints."""
    updated = deepcopy(calibration)
    index = updated['motor_names'].index('gripper')
    if updated['calib_mode'][index] != 'LINEAR':
        raise ValueError('Expected an existing LINEAR gripper calibration.')
    span = opened - closed
    old_span = updated['end_pos'][index] - updated['start_pos'][index]
    if not 10 <= abs(span) < 2048:
        raise ValueError('Unexpected endpoint span; check the poses and encoder for a whole-turn jump.')
    if span * old_span <= 0:
        raise ValueError('Gripper direction changed; check that closed and open poses were not swapped.')
    updated['start_pos'][index] = int(closed)
    updated['end_pos'][index] = int(opened)
    return updated


def sample_endpoint(bus):
    """Read raw positions; an unset bus calibration avoids the failing conversion."""
    if int(bus.read('Torque_Enable', 'gripper')[0]) != 0:
        raise RuntimeError('Gripper torque is enabled. Stop its controller before measuring; do not force it.')
    values = []
    for _ in range(20):
        values.append(int(bus.read('Present_Position', 'gripper')[0]))
        time.sleep(.02)
    if max(values) - min(values) > 10:
        raise RuntimeError('Gripper moved while sampling. Run again and hold it still at each endpoint.')
    return round(statistics.median(values))


def measure_endpoints(bus, calibration, arm, open_only=False):
    index = calibration['motor_names'].index('gripper')
    if open_only:
        closed = calibration['start_pos'][index]
        print(f'Keeping the existing closed position: {closed} ticks')
    else:
        input(f'Gently fully CLOSE the {arm} gripper, hold still, then press Enter: ')
        closed = sample_endpoint(bus)
        print(f'Closed: {closed} ticks')
    input(f'Move the {arm} gripper to the desired OPEN position, short of its mechanical stop; '
          'hold still, then press Enter: ')
    opened = sample_endpoint(bus)
    print(f'Open: {opened} ticks')
    return with_gripper_endpoints(calibration, closed, opened)


def save_calibration(path, original, updated):
    """Keep an exact backup and replace the JSON atomically."""
    if path.read_bytes() != original:
        raise RuntimeError('Calibration changed during measurement; no changes saved. Run again.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    backup = path.with_name(f'{path.name}.{stamp}.bak')
    with backup.open('xb') as stream:
        stream.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(updated, stream, indent=4)
            stream.write('\n')
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return backup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['leader', 'follower'], default='leader')
    parser.add_argument('--calibration', type=Path, help='Defaults to the selected right arm calibration.')
    parser.add_argument('--port', help='Defaults to the selected right arm serial port.')
    parser.add_argument('--open-only', action='store_true', help='Keep the saved closed position.')
    parser.add_argument('--save', action='store_true', help='Back up and save the measured endpoints.')
    args = parser.parse_args(argv)
    if args.calibration is None:
        args.calibration = DEFAULT_CALIBRATION.with_name(f'right_{args.arm}.json')
    if args.port is None:
        args.port = f'/dev/ttyDXL_{args.arm}_right'
    original = args.calibration.read_bytes()
    calibration = json.loads(original)
    # Validate the file before opening a device.
    index = calibration['motor_names'].index('gripper')
    with_gripper_endpoints(calibration, calibration['start_pos'][index], calibration['end_pos'][index])
    owner = subprocess.run(['fuser', args.port], capture_output=True, text=True)
    if owner.returncode == 0:
        raise RuntimeError(f'{args.arm.capitalize()} port is already in use by PID(s): {owner.stdout.strip()}')
    if owner.returncode != 1 or owner.stderr.strip():
        raise RuntimeError(f'Cannot check port ownership: {owner.stderr.strip()}')

    from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
    from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

    bus = DynamixelMotorsBus(DynamixelMotorsBusConfig(
        port=args.port, motors={'gripper': (9, 'xc430-w150' if args.arm == 'leader' else 'xm430-w350')},
    ))
    try:
        bus.connect()
        bus.set_bus_baudrate(1_000_000)
        print(f'Reading only the RIGHT {args.arm.upper()} gripper. No motor commands will be sent.')
        if int(bus.read('Torque_Enable', 'gripper')[0]) != 0:
            raise RuntimeError('Gripper torque is enabled. Stop its controller before measuring; do not force it.')
        updated = measure_endpoints(bus, calibration, args.arm, args.open_only)
    finally:
        if bus.is_connected:
            bus.disconnect()

    print(f"Gripper endpoints: {calibration['start_pos'][index]}–{calibration['end_pos'][index]}"
          f" -> {updated['start_pos'][index]}–{updated['end_pos'][index]} ticks")
    if args.save:
        backup = save_calibration(args.calibration, original, updated)
        print(f'Saved {args.calibration}\nBackup: {backup}')
    else:
        print('Preview only. Run again with --save to measure and save endpoints.')


if __name__ == '__main__':
    main()
