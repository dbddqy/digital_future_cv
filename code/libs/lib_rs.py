import pyrealsense2 as rs
import cv2
# import pclpy
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
        self.C_ext = np.array([[self.fx, 0., self.ppx, 0.],
                               [0., self.fy, self.ppy, 0.],
                               [0., 0., 1., 0.]], dtype=np.float32)
        # the camera from Shanghai
        self.ppx_r = 9.6591498e+02
        self.ppy_r = 5.4094659e+02
        self.fx_r = 1.3821929e+03
        self.fy_r = 1.3783481e+03
        self.C_r = np.array([[self.fx_r, 0., self.ppx_r],
                             [0., self.fy_r, self.ppy_r],
                             [0., 0., 1.]], dtype=np.float32)
        self.C_ext_r = np.array([[self.fx_r, 0., self.ppx_r, 0.],
                                 [0., self.fy_r, self.ppy_r, 0.],
                                 [0., 0., 1., 0.]], dtype=np.float32)
        self.coeffs_r = np.array([0., 0., 0., 0., 0.]).reshape([5, 1])
        # circle board related 01
        self.circles_size = (4, 11)
        circles_points = []
        for i in range(11):
            for j in range(4):
                circles_points.append([i*0.02, j*0.04+(i%2)*0.02, 0.0])
        self.circles_points = np.asarray(circles_points, dtype=np.float32)
        # circle board related 02
        self.circles_size_2 = (7, 7)
        circles_points_2 = []
        for i in range(7):
            for j in range(7):
                circles_points_2.append([i * 0.03, j * 0.03, 0.0])
        self.circles_points_2 = np.asarray(circles_points_2, dtype=np.float32)
        # aruco related
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # ==================================
    # 2d functionality
    # ==================================
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

    def detect_circle_board_2(self):
        color = self.get_frame_color()
        color_drawn = color.copy()
        is_found, centers = cv2.findCirclesGrid(color, self.circles_size_2, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        pose = np.eye(4, dtype=np.float32)
        if is_found:
            cv2.drawChessboardCorners(color_drawn, self.circles_size_2, centers, is_found)
            _, rvec, tvec = cv2.solvePnP(self.circles_points_2, centers, self.C, self.coeffs)
            cv2.aruco.drawAxis(color_drawn, self.C, self.coeffs, rvec, tvec, 0.054)
            # calculate pose
            pose[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
            pose[0:3, 3:4] = tvec
        return is_found, color, color_drawn, pose

    def detect_aruco(self):
        color = self.get_frame_color()
        color_drawn = color.copy()
        corners, ids, _ = cv2.aruco.detectMarkers(color_drawn, self.aruco_dict)
        cv2.aruco.drawDetectedMarkers(color_drawn, corners, ids)
        return corners, ids, color, color_drawn

    def detect_aruco_stuttgart(self):
        color = self.get_frame_color()
        color_drawn = color.copy()
        corners, ids, _ = cv2.aruco.detectMarkers(color_drawn, cv2.aruco.custom_dictionary(10, 6))
        cv2.aruco.drawDetectedMarkers(color_drawn, corners, ids)
        return corners, ids, color, color_drawn

    # ==================================
    # 3d functionality
    # ==================================
    def construct_point(self, u, v, d):
        point = np.zeros([3, ])
        point[0] = (u - self.ppx_r) * d / self.fx_r
        point[1] = (v - self.ppy_r) * d / self.fy_r
        point[2] = d
        return point

    def close(self):
        self.pipeline.stop()

# utility functions


def window_setting(config):
    if config == "color":
        cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("color", 960, 540)
    elif config == "color+depth":
        cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("color", 960, 540)
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("depth", 960, 540)


def aruco_dict():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


def coeffs():
    return np.array([0., 0., 0., 0., 0.]).reshape([5, 1])


def P(size, index):
    if index == 0:
        return np.array([[-0.5*size], [0.5*size], [0.], [1.]])
    if index == 1:
        return np.array([[0.5*size], [0.5*size], [0.], [1.]])
    if index == 2:
        return np.array([[0.5*size], [-0.5*size], [0.], [1.]])
    if index == 3:
        return np.array([[-0.5*size], [-0.5*size], [0.], [1.]])
