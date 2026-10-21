"""Shared right-follower serial I/O for tracking and automatic manipulation.

Importing this module opens no devices. Controllers take over an existing hold.
"""
import os
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from .vx300s_kinematics import MOTOR_IDS


class RightArm:
    def __init__(self, require_torque=True):
        import dynamixel_sdk as dxl
        self.dxl = dxl
        path = '/dev/ttyDXL_follower_right'
        owner = subprocess.run(['fuser',path],capture_output=True,text=True)
        if owner.returncode == 0:
            raise RuntimeError('Right follower serial port is already owned: '+owner.stdout.strip())
        self.port = dxl.PortHandler(path)
        self.packet = dxl.PacketHandler(2.)
        if not self.port.openPort():
            raise RuntimeError('Cannot open right follower port')
        try:
            if not self.port.setBaudRate(1_000_000):
                raise RuntimeError('Cannot set host serial baud rate')
            for motor in range(1,10):
                if require_torque and self.read_register(motor,64,1) != 1:
                    raise RuntimeError(f'Motor {motor} torque is off; no torque settings changed')
                if self.read_register(motor,70,1) != 0:
                    raise RuntimeError(f'Motor {motor} has a hardware error')
                if motor != 9 and self.read_register(motor,11,1) != 4:
                    raise RuntimeError('Arm operating mode differs from the audited extended-position mode')
            for shadow, master in [(3,2),(5,4)]:
                if self.read_register(shadow,12,1) != master:
                    raise RuntimeError('Paired-motor secondary ID is not configured as expected')
            expected_models = {i:1120 for i in range(1,8)} | {8:1020,9:1020}
            for motor, model in expected_models.items():
                if self.read_register(motor,0,2) != model:
                    raise RuntimeError('Motor model differs from audited right arm')
            self.reader = dxl.GroupSyncRead(self.port,self.packet,132,4)
            for motor in MOTOR_IDS:
                self.reader.addParam(motor)
            self.writer = dxl.GroupSyncWrite(self.port,self.packet,116,4)
        except BaseException:
            self.port.closePort()
            raise

    def read_register(self,motor,address,width):
        value,comm,error = getattr(self.packet,f'read{width}ByteTxRx')(self.port,motor,address)
        if comm != self.dxl.COMM_SUCCESS or error&127:
            raise RuntimeError(f'Motor {motor} register read failed: {comm}/{error}')
        if error&128:
            if address==70:
                kind='overload' if value&32 else 'hardware fault'
                raise RuntimeError(f'Motor {motor} {kind}: hardware error 0x{value:02x}; torque may be disabled')
            raise RuntimeError(f'Motor {motor} hardware alert during register {address} read; inspect hardware error and torque state')
        return value

    def positions(self):
        if self.reader.txRxPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Joint feedback failed')
        raw=[]
        for motor in MOTOR_IDS:
            if not self.reader.isAvailable(motor,132,4):
                raise RuntimeError('Incomplete joint feedback')
            value=self.reader.getData(motor,132,4)
            raw.append(value if value < 2**31 else value-2**32)
        return (np.array(raw)-2048)*2*np.pi/4096

    def goals(self):
        raw=[]
        for motor in MOTOR_IDS:
            value=self.read_register(motor,116,4)
            raw.append(value if value<2**31 else value-2**32)
        return (np.array(raw)-2048)*2*np.pi/4096

    def goal(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Invalid six-joint goal')
        raw=np.rint(2048+q*4096/(2*np.pi)).astype(int)
        self.writer.clearParam()
        for motor,value in zip(MOTOR_IDS,raw):
            value=int(value)&0xffffffff
            if not self.writer.addParam(motor,list(value.to_bytes(4,'little'))):
                raise RuntimeError('Could not build joint goal packet')
        if self.writer.txPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Goal packet failed')

    def close(self):
        self.port.closePort()


def signed(value, bits):
    return value - (1 << bits) if value >= (1 << (bits-1)) else value


class TapeArm(RightArm):
    def joint_velocity_limits_deg_s(self):
        """Read existing arm velocity/profile limits; never change registers."""
        limits = {i:self.read_register(i,44,4)*.229*6 for i in range(1,9)}
        for i in limits:
            profile = self.read_register(i,112,4)*.229*6
            if profile > 0:
                limits[i] = min(limits[i],profile)
        limits[2] = min(limits[2],limits[3])
        limits[4] = min(limits[4],limits[5])
        return [limits[i] for i in MOTOR_IDS]

    def __init__(self):
        super().__init__(require_torque=False)
        try:
            self.telemetry_reader = self.dxl.GroupSyncRead(self.port, self.packet, 126, 10)
            for motor in range(1, 10):
                if not self.telemetry_reader.addParam(motor):
                    raise RuntimeError('Cannot prepare telemetry reader')
                if self.read_register(motor, 10, 1) & 8:
                    raise RuntimeError('Unexpected torque-on-goal-update drive mode')
            if self.read_register(9, 11, 1) != 5:
                raise RuntimeError('Unexpected gripper operating mode')
        except BaseException:
            self.close()
            raise

    def telemetry(self):
        r = self.telemetry_reader
        if r.txRxPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Joint/current telemetry communication failed')
        raw, currents = {}, []
        for motor in range(1, 10):
            if not r.isAvailable(motor, 126, 10):
                raise RuntimeError('Incomplete motor telemetry')
            raw[motor] = signed(r.getData(motor, 132, 4), 32)
            currents.append(signed(r.getData(motor, 126, 2), 16)*.00269)
        q = (np.array([raw[i] for i in MOTOR_IDS])-2048)*2*np.pi/4096
        return q, np.array(currents), raw

    def check_health(self):
        for motor in range(1, 10):
            if self.read_register(motor, 70, 1) != 0:
                raise RuntimeError('Motor hardware fault')
            if self.read_register(motor, 64, 1) != 1:
                raise RuntimeError(f'Motor {motor} torque unexpectedly off')
