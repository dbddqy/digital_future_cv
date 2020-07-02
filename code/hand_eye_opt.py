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

c2o_all = np.loadtxt("..\\data_calib\\20200628_c2o.txt")
b2e_pos_all = np.loadtxt("..\\data_calib\\20200628_b2e_pos.txt") / 1000.
b2e_ori_all = np.loadtxt("..\\data_calib\\20200628_b2e_ori.txt")

b2e_ori_all = opt.swap_quats(b2e_ori_all)

# filter some bad data
skip_data = [2, 3, 9, 10, 15, 16, 17, 18, 19, 20, 25, 26, 28, 29, 30, 34, 35, 37, 38, 39, 40, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 69, 70, 79]
# skip_data = [2, 3, 16, 17, 30, 38, 39, 57]
# skip_data = []
num_data = c2o_all.shape[0] - len(skip_data)
c2o, b2e_pos, b2e_ori = [], [], []
for i in range(c2o_all.shape[0]):
    if i in skip_data:
        continue
    c2o.append(c2o_all[i])
    b2e_pos.append(b2e_pos_all[i])
    b2e_ori.append(b2e_ori_all[i])
c2o = np.array(c2o)
b2e_pos = np.array(b2e_pos)
b2e_ori = np.array(b2e_ori)
# shuffle data
shuffle_idx = np.arange(0, c2o.shape[0])
np.random.shuffle(shuffle_idx)
c2o = c2o[shuffle_idx]
b2e_pos = b2e_pos[shuffle_idx]
b2e_ori = b2e_ori[shuffle_idx]
# ------------------------
# prepare b2e
# ------------------------
b2e_ori_all = opt.swap_quats(b2e_ori_all)

b2e = np.zeros(c2o.shape)
for i in range(num_data):
    b2e[i] = opt.t_4_4__quat(b2e_ori[i], b2e_pos[i]).reshape([16, ])

# ------------------------
# load data ext
# ------------------------
c2o_ext = np.loadtxt("..\\data_calib\\20200625_c2o.txt")
b2e_pos_ext = np.loadtxt("..\\data_calib\\20200625_b2e_pos.txt") / 1000.
b2e_ori_ext = np.loadtxt("..\\data_calib\\20200625_b2e_ori.txt")

num_data_ext = c2o_ext.shape[0]
# shuffle data ext
shuffle_idx = np.arange(0, c2o_ext.shape[0])
np.random.shuffle(shuffle_idx)
c2o_ext = c2o_ext[shuffle_idx]
b2e_pos_ext = b2e_pos_ext[shuffle_idx]
b2e_ori_ext = b2e_ori_ext[shuffle_idx]

# ------------------------
# prepare b2e ext
# ------------------------
b2e_ori_ext = opt.swap_quats(b2e_ori_ext)

b2e_ext = np.zeros(c2o_ext.shape)
for i in range(c2o_ext.shape[0]):
    b2e_ext[i] = opt.t_4_4__quat(b2e_ori_ext[i], b2e_pos_ext[i]).reshape([16, ])

# ------------------------
# load data ext2
# ------------------------
c2o_ext2 = np.loadtxt("..\\data_calib\\20200628_c2o+.txt")
b2e_pos_ext2 = np.loadtxt("..\\data_calib\\20200628_b2e_pos+.txt") / 1000.
b2e_ori_ext2 = np.loadtxt("..\\data_calib\\20200628_b2e_ori+.txt")

# shuffle data ext
shuffle_idx = np.arange(0, c2o_ext2.shape[0])
np.random.shuffle(shuffle_idx)
c2o_ext2 = c2o_ext2[shuffle_idx]
b2e_pos_ext2 = b2e_pos_ext2[shuffle_idx]
b2e_ori_ext2 = b2e_ori_ext2[shuffle_idx]
# ------------------------
# prepare b2e ext
# ------------------------
b2e_ori_ext2 = opt.swap_quats(b2e_ori_ext2)

b2e_ext2 = np.zeros(c2o_ext2.shape)
for i in range(c2o_ext2.shape[0]):
    b2e_ext2[i] = opt.t_4_4__quat(b2e_ori_ext2[i], b2e_pos_ext2[i]).reshape([16, ])

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
for i in range(c2o_ext.shape[0]-1):
    A_whole.append(opt.inv(b2e_ext[i + 1].reshape([4, 4])).dot(b2e_ext[i].reshape([4, 4])))
    B_whole.append(c2o_ext[i + 1].reshape([4, 4]).dot(opt.inv(c2o_ext[i].reshape([4, 4]))))
for i in range(c2o_ext2.shape[0]-1):
    A_whole.append(opt.inv(b2e_ext2[i + 1].reshape([4, 4])).dot(b2e_ext2[i].reshape([4, 4])))
    B_whole.append(c2o_ext2[i + 1].reshape([4, 4]).dot(opt.inv(c2o_ext2[i].reshape([4, 4]))))

x_plot = range(140, len(A_whole))
y0_plot = []
y1_plot = []
y2_plot = []
y3_plot = []
y4_plot = []
y5_plot = []
for num in range(140, len(A_whole)):
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

