import cv2
import libs.lib_rs as rs

# generation
dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# for i in range(50):
#     image = cv2.aruco.drawMarker(dic, i, 2000)
#     cv2.imwrite(("..\\markers\\%d.png" % i), image)

# test detect
cv2.namedWindow("color", cv2.WINDOW_NORMAL)
cv2.resizeWindow("color", 960, 540)

d415 = rs.D415()

while cv2.waitKey(50) != ord('q'):
    color = d415.get_frame_color()
    corners, ids, _ = cv2.aruco.detectMarkers(color, dic)
    if len(corners) != 0:
        for i in range(len(corners)):
            color = cv2.aruco.drawDetectedMarkers(color, corners, ids)
    cv2.imshow("color", color)

d415.close()
