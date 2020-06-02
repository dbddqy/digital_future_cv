import pyrealsense2 as rs
import cv2
import numpy as np


class D415:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        cfg = self.pipeline.start(config)

        # set intrinsics
        profile = cfg.get_stream(rs.stream.color)
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        self.width = intrinsics.width
        self.height = intrinsics.height
        self.ppx = intrinsics.ppx
        self.ppy = intrinsics.ppy
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.coeffs = intrinsics.coeffs
        self.C = np.array([[self.fx, 0., self.ppx],
                           [0., self.fy, self.ppy],
                           [0., 0., 1.]])

    def get_frames(self):
        # align color frame to depth frame
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        color_rs = frames.get_color_frame()
        depth_rs = frames.get_depth_frame()

        color_mat = np.asanyarray(color_rs.as_frame().get_data())
        depth_mat = np.asanyarray(depth_rs.as_frame().get_data())
        return color_mat, depth_mat

    def close(self):
        self.pipeline.stop()
