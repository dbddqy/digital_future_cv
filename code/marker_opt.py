import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt

from matplotlib import pyplot as plt

# ------------------------
# solve: p = C * [I|0] * T(W->C) * T(K->W) * P(K)
# parameters to optimize:
# T(W->C) (6*n) n = 20
# T(K->W) (6*m) m = 4
# ------------------------

n = 20  # 20 photos
m = 4  # 4 markers

# ---------------------------------
# cost function: x: [c2w_rvec
#                    c2w_tvec
#                    ...(n times)
#                    w2k_rvec
#                    w2k_tvec
#                    ...(m-1 times)] # first one is identity
# ---------------------------------


def P(i, j, k):
    global ids_all
    marker_sizes = [0.2, 0.2, 0.09, 0.2]
    size = marker_sizes[ids_all[i][j, 0]]
    if k == 0:
        return np.array([-0.5*size, 0.5*size, 0., 1.]).reshape([4, 1])
    elif k == 1:
        return np.array([0.5*size, 0.5*size, 0., 1.]).reshape([4, 1])
    elif k == 2:
        return np.array([0.5*size, -0.5*size, 0., 1.]).reshape([4, 1])
    elif k == 3:
        return np.array([-0.5*size, -0.5*size, 0., 1.]).reshape([4, 1])


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
                corner = d415.C_ext.dot(c2w[i]).dot(w2k[ids[j, 0]]).dot(P(i, j, k))
                res[marker_index*8+k*2+0] = (corner[0][0] / corner[2][0] - corners[j][0, k][0])
                res[marker_index*8+k*2+1] = (corner[1][0] / corner[2][0] - corners[j][0, k][1])
            marker_index += 1
    return res


d415 = rs.D415()

# ------------------------
# load data
# ------------------------

path = "..\\data_2D\\marker_%d.png"
corners_all, ids_all = [], []
count_marker = 0
for i in range(20):
    color = cv2.imread(path % i)
    corners, ids, _ = cv2.aruco.detectMarkers(color, cv2.aruco.custom_dictionary(10, 6))
    # cv2.aruco.drawDetectedMarkers(color, corners, ids)
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
print(ls.x[6*n+6: 6*n+12])
print(ls.x[6*n+12: 6*n+18])
# print(ls.x[6*n+18: 6*n+24])
print("------------------")
print(ls.cost)
print(ls.optimality)
