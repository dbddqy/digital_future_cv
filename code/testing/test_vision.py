import cv2
# import numpy as np
import libs.lib_rs as rs


def window_setting():
    cv2.namedWindow("color", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("color", 960, 540)
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("depth", 960, 540)


window_setting()
d415 = rs.D415()

while cv2.waitKey(50) != ord('q'):
    color, depth = d415.get_frames()

    cv2.imshow("color", color)
    cv2.imshow("depth", depth)

d415.close()

