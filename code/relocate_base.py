import numpy as np
import cv2
import libs.lib_rs as rs
import libs.lib_optimization as opt

# ----------------
# load data
# ----------------
print("loading data...")

# c [3, 4] * 1
c = np.load("..\\data_locate_base\\c.npy")
print("c:")
print(c)

# c2e [4, 4] * 1
c2e = np.load("..\\data_locate_base\\c2e.npy")
print("c2e:")
print(c2e)

# e2b [4, 4] * number of photos
b2e_pos = np.loadtxt("..\\data_locate_base\\b2e_pos.txt") / 1000.
b2e_ori = np.loadtxt("..\\data_locate_base\\b2e_ori.txt")
e2b = opt.inv_t_4_4s__quats(opt.swap_quats(b2e_ori), b2e_pos)
n = len(e2b)
print("e2b:")
for i in range(len(e2b)):
    print(e2b[i])

# w2k [4, 4] * number of markers
w2k = opt.t_4_4s__rvecs6(np.load("..\\data_locate_base\\w2k.npy"))
print("w2k:")
for i in range(len(w2k)):
    print(w2k[i])
    np.savetxt(("w2k_%d.txt" % i), w2k[i].reshape([16, 1]))


# --------------------------------
# initial guess of b2w [6, ] * 1
# --------------------------------
print("initial guess of b2w:")
path_img = "..\\data_locate_base\\marker_%d.png"
size = 0.2
img = cv2.imread(path_img % 0)
corners, ids, _ = cv2.aruco.detectMarkers(img, rs.aruco_dict())
rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], size, c[0:3, 0:3], rs.coeffs())
c2k = opt.t_4_4__rvec(rvec.reshape([3, ]), tvec.reshape([3, ]))
x0 = opt.rvec6__t_4_4(opt.inv(e2b[0]).dot(opt.inv(c2e)).dot(c2k).dot(opt.inv(w2k[ids[0, 0]])))
print(x0)


def residual(x):
    global c, c2e, e2b, w2k, path_img, size, n
    res = []
    # i-th photo  j-th marker  k-th corner
    for i in range(n):
        img = cv2.imread(path_img % i)
        param = cv2.aruco.DetectorParameters_create()
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(img, rs.aruco_dict(), parameters=param)
        m = len(corners)  # num of markers
        b2w = opt.t_4_4__rvec6(x)
        for j in range(m):
            index = ids[j, 0]
            for k in range(4):
                p = c.dot(c2e).dot(e2b[i]).dot(b2w).dot(w2k[index]).dot(rs.P(size, k))
                res.append(p[0, 0] / p[2, 0] - corners[j][0, k, 0])
                res.append(p[1, 0] / p[2, 0] - corners[j][0, k, 1])
    return np.array(res)


ls = opt.least_squares(residual, x0, jac="2-point")
print("final b2w: ")
print(ls.x)
print("final cost: %f" % ls.cost)
np.savetxt("..\\data_locate_base\\result.txt", opt.inv(opt.t_4_4__rvec6(ls.x)).reshape([16, 1]))
# print(ls.optimality)

