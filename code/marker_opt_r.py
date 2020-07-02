import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt
from scipy.sparse import lil_matrix

# ------------------------
# solve: p = C * T(C->W) * T(W->K) * P(K)
# parameters to optimize:
# T(C->W) (6*n) n = 5
# T(W->K) (6*m) m = 2
# ------------------------

n = 280  # 280 photos
m = 53  # 53 markers
N = 264  # 251 photos to be used (264 for non-corner-opt)

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
    for i in range(N):
        c2w.append(opt.t_4_4__rvec(x[i*6: i*6+3], x[i*6+3: i*6+6]))
    w2k = []
    w2k.append(np.eye(4))
    for i in range(m-1):
        w2k.append(opt.t_4_4__rvec(x[6*N+i*6: 6*N+i*6+3], x[6*N+i*6+3: 6*N+i*6+6]))
    # residual
    res = np.zeros([count_marker*4*2, ])
    marker_index = 0
    # res = []
    # i-th photo  j-th marker  k-th corner
    for i in range(len(corners_all)):
        corners, ids = corners_all[i], ids_all[i]
        for j in range(len(corners)):
            # if ids[j, 0] != i and ids[j, 0] != i+1:
            #     continue
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

path = "..\\data_2D\\20200701_marker5\\marker_%d.png"
corners_all, ids_all = [], []
# indices = []
count_marker = 0
for i in range(280):
    color = cv2.imread(path % i)
    if color is None:
        continue
    param = cv2.aruco.DetectorParameters_create()
    # param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    # if i == 17:
    #     param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    corners, ids, _ = cv2.aruco.detectMarkers(color, rs.aruco_dict(), parameters=param)
    if len(corners) != 2:
        continue
    # cv2.aruco.drawDetectedMarkers(color, corners, ids)
    # cv2.imshow("color", color)
    # cv2.waitKey()
    corners_all.append(corners)
    ids_all.append(ids)
    count_marker += len(ids)

print(count_marker)

# c2w = np.load("..\\data_2D\\c2w.npy").flatten()
w2k = np.load("..\\data_2D\\w2k.npy")
w2k_4_4 = opt.t_4_4s__rvecs6(w2k)
c2w = []
for i in range(len(ids_all)):
    ids = ids_all[i]
    corners = corners_all[i]
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.2, d415.C_r, d415.coeffs_r)
    c2k = opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ]))
    c2w.append(opt.rvec6__t_4_4(c2k.dot(opt.inv(w2k_4_4[ids[0, 0]]))))
c2w = np.array(c2w)
c2w = c2w.flatten()
w2k = w2k[1:].flatten()

x0 = np.hstack((c2w, w2k))


A = lil_matrix((N*16, N*6+(m-1)*6), dtype=int)
for i in range(N):
    A[np.arrange(i*16, i*16+16), np.arrange(i*6, i*6+6)] = 1
    index_m0 = ids_all[i][0, 0] - 1
    index_m1 = ids_all[i][1, 0] - 1
    if index_m0 != -1:
        A[np.arrange(i*16, i*16+8), np.arrange(6*N+index_m0*6, 6*N+index_m0*6+6)] = 1
    if index_m1 != -1:
        A[np.arrange(i*16+8, i*16+16), np.arrange(6*N+index_m1*6, 6*N+index_m1*6+6)] = 1


# ls = opt.least_squares(residual, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
ls = opt.least_squares(residual, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf')

print(ls.x[6*N: 6*N+6])
print(ls.x[6*N+6: 6*N+12])

print("------------------")
print(ls.cost)
print(ls.optimality)

final_w2k = []
final_w2k.append(opt.rvec6__t_4_4(np.eye(4)))
for i in range(m-1):
    final_w2k.append(ls.x[6*(N+i): 6*(N+i+1)])
np.save("w2k", final_w2k)
