import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt

path = "..\\data_2D\\marker_%d.png"
d415 = rs.D415()

# ------------------------
# initial guesses
# ------------------------
w2k = [np.eye(4)]*4
w2k[0] = np.eye(4)
w2k[1] = opt.t_4_4__rvec(np.array([-0.01162871, 0.02554612, 0.01882586]),
                        np.array([-0.48797692, 0.04626508, -0.0053757]))
w2k[2] = opt.t_4_4__rvec(np.array([-0.00543519, -0.02607094, -2.00999557]),
                        np.array([-1.3201429, 0.4008036, 0.00386429]))
w2k[3] = opt.t_4_4__rvec(np.array([-0.01794776, 0.00204748, -0.30005135]),
                        np.array([-0.95785263, 0.12048876, 0.00879366]))
np.save("w2k", opt.rvecs6__t_4_4s(w2k[1:]))  # exclude
# one

# -------------------------
# for camera poses
# -------------------------

c2w = np.zeros([20, 6])
for i in range(20):
    color = cv2.imread(path % i)
    corners, ids, _ = cv2.aruco.detectMarkers(color, cv2.aruco.custom_dictionary(10, 6))
    index = ids[0, 0]
    if index == 2:
        size = 0.09
    else:
        size = 0.2
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], size, d415.C, d415.coeffs)
    c2k = opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ]))
    c2w[i] = opt.rvec6__t_4_4(c2k.dot(opt.inv(w2k[index])))
print(c2w)
np.save("c2w", c2w)
# -------------------------
# for marker relative poses
# -------------------------

# color = cv2.imread(path % 12)
# corners, ids, _ = cv2.aruco.detectMarkers(color, cv2.aruco.custom_dictionary(10, 6))
#
# c2k = []
# for i in range(2):
#     print(ids[i])
#     if ids[i, 0] == 2:
#         size = 0.09
#     else:
#         size = 0.2
#     rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size, d415.C, d415.coeffs)
#     print("t")
#     print(tvec)
#     print("r")
#     print(rvec)
#     c2k.append(opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ])))
#
# print("================")
# k2 = opt.rvec__t_4_4(ks[3].dot(opt.inv(c2k[1])).dot(c2k[0]))
# print(k2)