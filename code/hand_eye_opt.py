import numpy as np
import libs.lib_optimization as opt

from matplotlib import pyplot as plt

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

c2o = np.loadtxt("..\\data_calib\\20200625_c2o.txt")
b2e_pos = np.loadtxt("..\\data_calib\\20200625_b2e_pos.txt") / 1000.
b2e_ori = np.loadtxt("..\\data_calib\\20200625_b2e_ori.txt")

b2e_ori = opt.swap_quats(b2e_ori)

num_data = c2o.shape[0]

# ------------------------
# prepare b2e
# ------------------------

b2e = np.zeros(c2o.shape)
for i in range(num_data):
    b2e[i] = opt.t_4_4__quat(b2e_ori[i], b2e_pos[i]).reshape([16, ])
# print(c2o[46].reshape([16, 1]))

for i in range(16):
    print(b2e[14, i])

# print("=========================")
# for i in range(num_data):
#     for j in range(16):
#         print(b2e[i, j], end="")
#         print(" ", end="")
#     print("")
# print("=========================")

# -------------------------
# solve AX = XB
# A = T2(B_E).inv * T1(B_E)
# B = T2(C_O) * T2(C_O).inv
# -------------------------

A_whole, B_whole = [], []
A, B = [], []

for i in range(num_data-1):
    A_whole.append(opt.inv(b2e[i + 1].reshape([4, 4])).dot(b2e[i].reshape([4, 4])))
    B_whole.append(c2o[i + 1].reshape([4, 4]).dot(opt.inv(c2o[i].reshape([4, 4]))))

x_plot = range(5, num_data)
y0_plot = []
y1_plot = []
y2_plot = []
y3_plot = []
y4_plot = []
y5_plot = []
for num in range(5, num_data):
    A = A_whole[0:num]
    B = B_whole[0:num]
    x0_mat = np.array([[-0.46491281, 0.88355481, 0.05645332, 0.04465462],
                       [-0.88535499, -0.4638495, -0.03146717, 0.04170271],
                       [-0.00161712, -0.06461072, 0.99790923, 0.156906],
                       [0., 0., 0., 1.]])
    x0 = opt.rvec6__t_4_4(x0_mat)
    ls = opt.least_squares(residual, x0, jac="3-point")
    y0_plot.append(ls.x[0])
    y1_plot.append(ls.x[1])
    y2_plot.append(ls.x[2])
    y3_plot.append(ls.x[3])
    y4_plot.append(ls.x[4])
    y5_plot.append(ls.x[5])

plt.plot(x_plot, y0_plot)
plt.show()
plt.plot(x_plot, y1_plot)
plt.show()
plt.plot(x_plot, y2_plot)
plt.show()
plt.plot(x_plot, y3_plot)
plt.show()
plt.plot(x_plot, y4_plot)
plt.show()
plt.plot(x_plot, y5_plot)
plt.show()

print(ls.x)

np.save("c2e", opt.inv(opt.t_4_4__rvec6(ls.x)))

print(ls.cost)
print(ls.optimality)

