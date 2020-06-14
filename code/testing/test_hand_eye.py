import numpy as np
import libs.lib_optimization as opt


# ------------------------
# cost function: x: [rvec
#                    tvec]
# ------------------------

def residual(x):
    global A, B
    res = np.zeros([16, ])
    x_mat = opt.t_4_4__rvec(x[0:3], x[3:6])
    for i in range(len(A)):
        res += np.power( A[i].dot(x_mat) - x_mat.dot(B[i]) , 2).reshape([16, ])
    return res


# ------------------------
# load data
# ------------------------

c2o = np.loadtxt("..\\..\\data_calib\\20200611_c2o_old.txt")
b2e = np.loadtxt("..\\..\\data_calib\\20200611_w2e_old.txt")

for i in range(b2e.shape[0]):
    if (abs(b2e[i][3]) > 50.0):
        b2e[i][3] *= 0.001
        b2e[i][7] *= 0.001
        b2e[i][11] *= 0.001

num_data = c2o.shape[0]

# ------------------------
# prepare b2e
# ------------------------

# -------------------------
# solve AX = XB
# A = T2(B_E).inv * T1(B_E)
# B = T2(C_O) * T2(C_O).inv
# -------------------------

A, B = [], []
for i in range(num_data-1):
    A.append(opt.inv(b2e[i+1].reshape([4, 4])).dot(b2e[i].reshape([4, 4])))
    B.append(c2o[i+1].reshape([4, 4]).dot(opt.inv(c2o[i].reshape([4, 4]))))

x0_mat = np.array([[-0.5, 0.866, 0., 0.020806],
                   [-0.866, -0.5, 0., 0.010413],
                   [0., 0., 1., 0.160870],
                   [0., 0., 0., 1.]])
# x0_mat = np.array([[0., -0.5, 0.866, 0.160870],
#                    [0., -0.866, -0.5, 0.020806],
#                    [1., 0., 0., 0.010413],
#                    [0., 0., 0., 1.]])
x0 = opt.rvec__t_4_4(x0_mat)
print(x0)
ls = opt.least_squares(residual, x0, jac="3-point")

print(ls.x)
print(ls.cost)
print(ls.optimality)
