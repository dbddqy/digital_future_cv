import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt

path = "..\\data_2D\\20200621_marker\\marker_%d.png"
d415 = rs.D415()

# ------------------------
# initial guesses
# ------------------------

w2k = [np.eye(4)]*2
w2k[0] = np.eye(4)
w2k[1] = opt.t_4_4__rvec(np.array([0.01169517, -0.01679191, -0.01882297]),
                        np.array([0.39514316, -0.01755504, 0.00154338]))
# np.save("w2k", opt.rvecs6__t_4_4s(w2k[1:]))  # exclude first one

# -------------------------
# for camera poses
# -------------------------

n = 5  # number of photos
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

# color = cv2.imread(path % 0)
# corners, ids, _ = cv2.aruco.detectMarkers(color, d415.aruco_dict)
#
# c2k = []
# for i in range(2):
#     print(ids[i])
#     size = 0.2
#     rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size, d415.C_r, d415.coeffs_r)
#     print("t")
#     print(tvec)
#     print("r")
#     print(rvec)
#     c2k.append(opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ])))
#
# print("================")
# k1 = opt.inv(c2k[0]).dot(c2k[1])
# print(opt.rvec6__t_4_4(k1))

d415.close()
