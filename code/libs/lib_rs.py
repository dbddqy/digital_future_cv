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
        self.coeffs = np.asarray(intrinsics.coeffs, dtype=np.float32).reshape([5, 1])
        self.C = np.array([[self.fx, 0., self.ppx],
                           [0., self.fy, self.ppy],
                           [0., 0., 1.]], dtype=np.float32)

        # circle board related
        self.circles_size = (4, 11)
        circles_points = []
        for i in range(11):
            for j in range(4):
                circles_points.append([i*0.02, j*0.04+(i%2)*0.02, 0.0])
        self.circles_points = np.asarray(circles_points, dtype=np.float32)

    def get_frames(self):
        # align color frame to depth frame
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        color_rs = frames.get_color_frame()
        depth_rs = frames.get_depth_frame()

        color_mat = np.asanyarray(color_rs.as_frame().get_data())
        depth_mat = np.asanyarray(depth_rs.as_frame().get_data())
        return color_mat, depth_mat

    def get_frame_color(self):
        frames = self.pipeline.wait_for_frames()
        color_rs = frames.get_color_frame()
        color_mat = np.asanyarray(color_rs.as_frame().get_data())
        return color_mat

    def detect_circle_board(self):
        color = self.get_frame_color()
        color_drawn = color.copy()
        is_found, centers = cv2.findCirclesGrid(color, self.circles_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        pose = np.eye(4, dtype=np.float32)
        if is_found:
            cv2.drawChessboardCorners(color_drawn, self.circles_size, centers, is_found)
            _, rvec, tvec = cv2.solvePnP(self.circles_points, centers, self.C, self.coeffs)
            cv2.aruco.drawAxis(color_drawn, self.C, self.coeffs, rvec, tvec, 0.12)
            # calculate pose
            pose[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
            pose[0:3, 3:4] = tvec
        return is_found, color, color_drawn, pose

    def close(self):
        self.pipeline.stop()
