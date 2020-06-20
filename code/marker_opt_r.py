import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt

# ------------------------
# solve: p = C * T(C->W) * T(W->K) * P(K)
# parameters to optimize:
# T(C->W) (6*n) n = 5
# T(W->K) (6*m) m = 2
# ------------------------

n = 5  # 5 photos
m = 2  # 2 markers

# ---------------------------------
# cost function: x: [c2w_rvec
#                    c2w_tvec
#                    ...(n times)
#                    w2k_rvec
#                    w2k_tvec
#                    ...(m-1 times)] # first one is identity
# ---------------------------------


def residual(x):
    global corners_all, ids_all, count_marker
    c2w = []
    for i in range(n):
        c2w.append(opt.t_4_4__rvec(x[i*6: i*6+3], x[i*6+3: i*6+6]))
    w2k = []
    w2k.append(np.eye(4))
    for i in range(m-1):
        w2k.append(opt.t_4_4__rvec(x[6*n+i*6: 6*n+i*6+3], x[6*n+i*6+3: 6*n+i*6+6]))
    # residual
    res = np.zeros([count_marker*4*2, ])
    marker_index = 0
    # res = []
    # i-th photo  j-th marker  k-th corner
    for i in range(len(corners_all)):
        corners, ids = corners_all[i], ids_all[i]
        for j in range(len(corners)):
            for k in range(4):
                corner = d415.C_ext_r.dot(c2w[i]).dot(w2k[ids[j, 0]]).dot(rs.P(0.2, k))
                res[marker_index*8+k*2+0] = (corner[0][0] / corner[2][0] - corners[j][0, k][0])
                res[marker_index*8+k*2+1] = (corner[1][0] / corner[2][0] - corners[j][0, k][1])
            marker_index += 1
    return res


d415 = rs.D415()

# ------------------------
# load data
# ------------------------

path = "..\\data_2D\\20200621_marker\\marker_%d.png"
corners_all, ids_all = [], []
count_marker = 0
for i in range(n):
    color = cv2.imread(path % i)
    param = cv2.aruco.DetectorParameters_create()
    param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(color, rs.aruco_dict(), parameters=param)
    # if i == 0:
    #     cv2.aruco.drawDetectedMarkers(color, corners, ids)
    #     cv2.imwrite("test_sub.png", color)
    # cv2.imshow("color", color)
    # cv2.waitKey()
    corners_all.append(corners)
    ids_all.append(ids)
    count_marker += len(ids)

print(count_marker)

# print(corners_all[0][2][0, 1])
# print(ids_all[0][2])

c2w = np.load("..\\data_2D\\c2w.npy").flatten()
w2k = np.load("..\\data_2D\\w2k.npy").flatten()

x0 = np.hstack((c2w, w2k))

# x0 = [0., 3.14, 0., 0., 0., 1.] * n
# # x0.extend([0., 0., 6.28, 0., 0., 0.])
# x0.extend([0., 0., 3.14, 0.48, -0.05, 0.])
# x0.extend([0., 0., 1.57, 0.42, 1.23, 0.])
# x0.extend([0., 0., 3.14, -0.98, 0.16, 0.])
# x0 = np.array(x0)
ls = opt.least_squares(residual, x0, jac="2-point")

print(ls.x[6*n: 6*n+6])
# print(ls.x[6*n+18: 6*n+24])
print("------------------")
print(ls.cost)
print(ls.optimality)
