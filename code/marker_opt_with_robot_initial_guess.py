import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt

b2e_pos = np.loadtxt("..\\data_calib\\20200617_b2e_pos.txt") / 1000.
b2e_ori = np.loadtxt("..\\data_calib\\20200617_b2e_ori.txt")

b2e = opt.t_4_4s__quats(opt.swap_quats(b2e_ori), b2e_pos)

e2c_rvec = np.loadtxt("..\\data_calib\\20200612_result\\final_result.txt")
e2c = opt.t_4_4__rvec(e2c_rvec[0:3], e2c_rvec[3:6])

# ---------------
# load image data
# ---------------
d415 = rs.D415()
c2k_0 = []
c2k_1 = []
img_path = "..\\data_2D\\20200617_marker\\marker_%d.png"
for i in range(5):
    img = cv2.imread(img_path % i)
    corners, ids, _ = cv2.aruco.detectMarkers(img, d415.aruco_dict)
    for j in range(len(corners)):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[j], 0.2, d415.C_r, d415.coeffs_r)
        # cv2.aruco.drawAxis(img, d415.C_r, d415.coeffs_r, rvec, tvec, 0.2)
        c2k = opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ]))
        if ids[j] == 0:
            c2k_0.append(c2k)
        elif ids[j] == 1:
            c2k_1.append(c2k)
    # cv2.imshow("color", img)
    # cv2.waitKey()

for i in range(5):
    b2k_0 = b2e[i].dot(e2c).dot(c2k_0[i])
    # print("b2k_0: ")
    # print(b2k_0)
    b2k_1 = b2e[i].dot(e2c).dot(c2k_1[i])
    # print("b2k_1: ")
    # print(b2k_1)
    print("1 to 2:")
    print(opt.inv(b2k_0).dot(b2k_1))
