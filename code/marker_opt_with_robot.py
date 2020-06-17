import numpy as np
import libs.lib_rs as rs
import cv2
import libs.lib_optimization as opt

# -----------------
# load poses
# -----------------
b2e_pos = np.loadtxt("..\\data_calib\\20200617_b2e_pos.txt") / 1000.
b2e_ori = np.loadtxt("..\\data_calib\\20200617_b2e_ori.txt")

# swap quaternions
e2b = opt.inv_t_4_4s__quats(opt.swap_quats(b2e_ori), b2e_pos)

e2c_rvec = np.loadtxt("..\\data_calib\\20200612_result\\final_result.txt")
c2e = opt.inv(opt.t_4_4__rvec(e2c_rvec[0:3], e2c_rvec[3:6]))
# np.save("c2e", c2e)

# ---------------
# load image data
# ---------------
d415 = rs.D415()
marker_size = 0.2
p = [[], []]
img_path = "..\\data_2D\\20200617_marker\\marker_%d.png"
for i in range(5):
    img = cv2.imread(img_path % i)
    corners, ids, _ = cv2.aruco.detectMarkers(img, d415.aruco_dict)
    for j in range(len(corners)):
        p[ids[j, 0]].append(corners[j].reshape([4, 2]))

# ---------------
# initial guess
# ---------------
x0 = np.zeros([12, ])
x0[0:6] = opt.rvec6__t_4_4(np.array([[0.99486095, 0.10078919, 0.00965603, 0.30825997],
                                     [-0.10085231, 0.99488154, 0.00628804, 1.63810419],
                                     [-0.00897284, -0.00722956, 0.99993361, -0.55456868],
                                     [0., 0., 0., 1.]]))
x0[6:12] = opt.rvec6__t_4_4(np.array([[-0.99637026, -0.0836713, -0.01566585, 0.62051195],
                                      [0.08371704, -0.99648695, -0.0022858,   1.60664948],
                                      [-0.01541956, -0.003589,   0.99987467, -0.55516513],
                                      [0.,          0.,          0.,          1.]]))

# ------------------------
# cost function: x: [k2b[0]
#                    k2b[1]]
# ------------------------


def P(k):
    global marker_size
    if k == 0:
        return np.array([[-0.5*marker_size], [0.5*marker_size], [0.], [1.]])
    if k == 1:
        return np.array([[0.5*marker_size], [0.5*marker_size], [0.], [1.]])
    if k == 2:
        return np.array([[0.5*marker_size], [-0.5*marker_size], [0.], [1.]])
    if k == 3:
        return np.array([[-0.5*marker_size], [-0.5*marker_size], [0.], [1.]])


def residual(x):
    global e2b, c2e, d415, p
    # res = np.zeros([2, ])
    res = []
    b2k = []
    b2k.append(opt.t_4_4__rvec(x[0:3], x[3:6]))
    b2k.append(opt.t_4_4__rvec(x[6:9], x[9:12]))
    # i-th photo  j-th marker  k-th corner
    for i in range(5):
        for j in range(2):
            for k in range(4):
                corner = d415.C_ext_r.dot(c2e).dot(e2b[i]).dot(b2k[j]).dot(P(k))
                res.append(corner[0, 0] / corner[2, 0] - p[j][i][k, 0])
                res.append(corner[1, 0] / corner[2, 0] - p[j][i][k, 1])
    return np.array(res)


ls = opt.least_squares(residual, x0, jac="2-point")

b2k_0 = opt.t_4_4__rvec(ls.x[0:3], ls.x[3:6])
b2k_1 = opt.t_4_4__rvec(ls.x[6:9], ls.x[9:12])
w2k_01 = opt.inv(b2k_0).dot(b2k_1)
# print(opt.rvec__t_4_4(w2k_01))
print(ls.cost/40.0)
print(ls.optimality)

w2k = []
w2k.append(opt.rvec6__t_4_4(np.eye(4)))
w2k.append(opt.rvec6__t_4_4(w2k_01))
print(w2k)
np.save("w2k", w2k)
