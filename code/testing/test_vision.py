import cv2
# import numpy as np
import libs.lib_rs as rs


# def window_setting():
#     cv2.namedWindow("color", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("color", 960, 540)
#     cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("depth", 960, 540)
#
#
# window_setting()
# d415 = rs.D415()
#
# while cv2.waitKey(50) != ord('q'):
#     color, depth = d415.get_frames()
#
#     cv2.imshow("color", color)
#     cv2.imshow("depth", depth)
#
# d415.close()

img = cv2.imread("circle.png")
# cv2.imshow("img", img)
# cv2.waitKey()
params = cv2.SimpleBlobDetector_Params()
params.maxArea = 100000
detector = cv2.SimpleBlobDetector_create(params)
is_found, centers = cv2.findCirclesGrid(img, (2, 13), flags=(cv2.CALIB_CB_ASYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING), blobDetector=detector)
print (is_found)
if is_found:
    cv2.drawChessboardCorners(img, (2, 13), centers, is_found)

cv2.imwrite("result.jpg", img)
