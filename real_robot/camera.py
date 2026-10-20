"""RealSense adapter, imported only for hardware commands. Does not reset devices."""
import time
import importlib.metadata
import numpy as np


def enumerate_cameras():
    import pyrealsense2 as rs
    result = []
    for device in rs.context().query_devices():
        result.append({key: device.get_info(getattr(rs.camera_info, key))
                       for key in ['name', 'serial_number', 'firmware_version', 'usb_type_descriptor']
                       if device.supports(getattr(rs.camera_info, key))})
    return result


class RGBDCamera:
    def __init__(self, serial, width=640, height=480, fps=15):
        import pyrealsense2 as rs
        self.rs, self.serial = rs, serial
        self.frames_started = False
        self.pipeline, config = rs.pipeline(), rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        try:
            self.scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
        except Exception:
            self.pipeline.stop()
            raise

    def read(self):
        # Startup can yield an old color frame paired with a new depth frame.
        # Wait briefly for a valid pair, never increase the permitted skew.
        wait_budget = 3.0 if self.frames_started else 10.0
        deadline = time.monotonic() + wait_budget
        skipped = timeouts = 0
        while True:
            arrived, frames = self.pipeline.try_wait_for_frames(1000)
            if not arrived:
                timeouts += 1
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Camera did not provide synchronized frames within {wait_budget:g}s')
                continue
            arrival = time.monotonic()
            aligned = self.align.process(frames)
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            paired = bool(color and depth)
            same_clock = paired and color.get_frame_timestamp_domain() == depth.get_frame_timestamp_domain()
            skew = abs(color.get_timestamp() - depth.get_timestamp()) / 1000 if same_clock else float('inf')
            if same_clock and skew <= 0.04:
                self.frames_started = True
                break
            skipped += 1
            if time.monotonic() >= deadline:
                raise RuntimeError(f'No synchronized RGB/depth pair within {wait_budget:g}s; last skew {skew:.3f}s')
        intr = depth.profile.as_video_stream_profile().get_intrinsics()
        metadata = {
            'camera_serial': self.serial,
            'pyrealsense2_version': importlib.metadata.version('pyrealsense2'),
            'sensor_timestamp_s': depth.get_timestamp() / 1000,
            'color_timestamp_s': color.get_timestamp() / 1000,
            'timestamp_domain': str(depth.get_frame_timestamp_domain()),
            'host_arrival_monotonic_s': arrival,
            'rgb_depth_skew_s': skew,
            'skipped_unsynchronized_pairs': skipped,
            'capture_wait_timeouts': timeouts,
            'depth_frame_number': depth.get_frame_number(),
            'color_frame_number': color.get_frame_number(),
            'depth_scale_m': self.scale,
            'intrinsics': {k: getattr(intr, k) for k in ['width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs']},
            'distortion_model': str(intr.model),
        }
        # RealSense deprojection accounts for the stream's distortion model.
        deproject = lambda pixel, z: self.rs.rs2_deproject_pixel_to_point(intr, pixel, z)
        return (np.asanyarray(color.get_data()).copy(),
                np.asanyarray(depth.get_data()).astype(float) * self.scale,
                metadata, deproject)

    def close(self):
        self.pipeline.stop()
