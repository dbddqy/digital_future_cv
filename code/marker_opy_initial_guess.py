import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt

path = "..\\data_2D\\20200628_marker\\marker_%d.png"
d415 = rs.D415()

# ------------------------
# initial guesses
# ------------------------

w2k = [np.eye(4)]*3
w2k[0] = np.eye(4)
w2k[1] = opt.t_4_4__rvec(np.array([1.07888743e-02, -3.80953256e-02, -1.59157464e+00]),
                         np.array([-9.08505140e-03, -4.23919083e-01, -1.34970647e-04]))
w2k[2] = opt.t_4_4__rvec(np.array([-2.29909471e-02, 1.96608912e-02, -1.56549568e+00]),
                         np.array([6.33976050e-04, 3.89877312e-01, -5.58933275e-03]))
np.save("w2k", opt.rvecs6__t_4_4s(w2k[1:]))  # exclude first one

# -------------------------
# for camera poses
# -------------------------

n = 9  # number of photos
c2w = np.zeros([n, 6])
for i in range(n):
    color = cv2.imread(path % i)
    corners, ids, _ = cv2.aruco.detectMarkers(color, d415.aruco_dict)
    index = ids[0, 0]
    size = 0.2
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], size, d415.C_r, d415.coeffs_r)
    c2k = opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ]))
    c2w[i] = opt.rvec6__t_4_4(c2k.dot(opt.inv(w2k[index])))
print(c2w)
np.save("c2w", c2w)

# -------------------------
# for marker relative poses
# -------------------------

# color = cv2.imread(path % 5)
# gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
# param = cv2.aruco.DetectorParameters_create()
# corners, ids, _ = cv2.aruco.detectMarkers(color, d415.aruco_dict)
# # corner refinement
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
# for i in range(len(corners)):
#     cv2.cornerSubPix(gray, corners[1], (3, 3), (-1, -1), criteria)
#
# cv2.aruco.drawDetectedMarkers(color, corners, ids)
# cv2.imshow("color", color)
# cv2.waitKey()
#
# c2k = []
# for i in range(2):
#     print(ids[i])
#     size = 0.2
#     rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size, d415.C_r, d415.coeffs_r)
#     # cv2.aruco.drawAxis(color, d415.C_r, d415.coeffs_r, rvec, tvec, 0.2)
#     # cv2.imshow("color", color)
#     # cv2.waitKey()
#     print("t")
#     print(tvec)
#     print("r")
#     print(rvec)
#     c2k.append(opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ])))
#
# print("================")
# k1 = opt.inv(c2k[1]).dot(c2k[0])
# print(opt.rvec6__t_4_4(k1))

d415.close()
