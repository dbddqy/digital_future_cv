import cv2
import numpy as np
import libs.lib_rs as rs

path = "..\\data_2D\\circles\\circles_%d.png"
d415 = rs.D415()
poses_str = ""

for i in range(80):
    color = cv2.imread(path % i)
    is_found, centers = cv2.findCirclesGrid(color, d415.circles_size_2, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    pose = np.eye(4, dtype=np.float32)
    if is_found:
        # cv2.drawChessboardCorners(color, d415.circles_size_2, centers, is_found)
        _, rvec, tvec = cv2.solvePnP(d415.circles_points_2, centers, d415.C_r, d415.coeffs_r)
        # cv2.aruco.drawAxis(color, d415.C_r, d415.coeffs_r, rvec, tvec, 0.21)
        # calculate pose
        pose[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
        pose[0:3, 3:4] = tvec
        for v in range(4):
            for u in range(4):
                poses_str += ("%f " % pose[v, u])
        poses_str += '\n'

print(poses_str)
d415.close()
