"""Disable right-follower torque only after the arm is physically supported."""
import argparse
import json
import subprocess


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supported',action='store_true',help='Confirm the arm is securely supported')
    parser.add_argument('--execute',action='store_true',help='Actually disable torque; otherwise read only')
    args=parser.parse_args(argv)
    if args.execute and not args.supported:
        parser.error('Support the arm first, then supply --supported --execute')
    import dynamixel_sdk as dxl
    owner=subprocess.run(['fuser','/dev/ttyDXL_follower_right'],capture_output=True,text=True)
    if owner.returncode==0:raise RuntimeError('Stop the current right-arm serial owner first: '+owner.stdout.strip())
    port=dxl.PortHandler('/dev/ttyDXL_follower_right');packet=dxl.PacketHandler(2.0)
    if not port.openPort():raise RuntimeError('Cannot open right-follower port')
    try:
        if not port.setBaudRate(1000000):raise RuntimeError('Cannot set baud rate')
        def read(motor,address):
            value,result,error=packet.read1ByteTxRx(port,motor,address)
            if result!=dxl.COMM_SUCCESS or error:raise RuntimeError(f'Motor {motor} read failed: {result}/{error}')
            return value
        before={i:read(i,64) for i in range(1,10)}
        if args.execute:
            # The user must support the complete arm before this write.
            group=dxl.GroupSyncWrite(port,packet,64,1)
            for i in range(1,10):
                if not group.addParam(i,[0]):raise RuntimeError('Cannot build torque-off packet')
            if group.txPacket()!=dxl.COMM_SUCCESS:raise RuntimeError('Torque-off communication failed')
        after={i:read(i,64) for i in range(1,10)}
        print(json.dumps({'executed':args.execute,'torque_before':before,'torque_after':after},indent=2))
        if args.execute and any(after.values()):raise RuntimeError('Torque-off verification failed')
    finally:port.closePort()


if __name__=='__main__':main()
