import numpy as np
import libs.lib_frame as f
import yaml

from matplotlib import pyplot as plt

# ------------------------
# cost function: x: [rvec
#                    tvec]
# ------------------------

def residual(x):
    global A, B
    res = np.zeros([12 * len(A), ])
    x_mat = f.t_4_4__rvec(x[0:3], x[3:6])
    for i in range(len(A)):
        res[12*i:12*(i+1)] = ( A[i].dot(x_mat) - x_mat.dot(B[i]) )[0:3, 0:4].reshape([12, ])
    return res


# ------------------------
# load data
# ------------------------
with open("config\\config_hand_eye.yml", 'r') as file:
    conf = yaml.safe_load(file.read())

c2o_all = np.loadtxt(conf["c2o_path"])
b2e_pos_all = np.loadtxt(conf["b2e_pos_path"]) / 1000.
b2e_ori_all = np.loadtxt(conf["b2e_ori_path"])

b2e_ori_all = f.swap_quats(b2e_ori_all)

# skip some bad data
skip_data = conf["skip_indices"]
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
b2e_ori_all = f.swap_quats(b2e_ori_all)

b2e = np.zeros(c2o.shape)
for i in range(num_data):
    b2e[i] = f.t_4_4__quat(b2e_ori[i], b2e_pos[i]).reshape([16, ])

# -------------------------
# solve AX = XB
# A = T2(B_E).inv * T1(B_E)
# B = T2(C_O) * T2(C_O).inv
# -------------------------

A, B = [], []

for i in range(num_data-1):
    A.append(f.inv(b2e[i + 1].reshape([4, 4])).dot(b2e[i].reshape([4, 4])))
    B.append(c2o[i + 1].reshape([4, 4]).dot(f.inv(c2o[i].reshape([4, 4]))))

x0_mat = np.array([[-0.46491281, 0.88355481, 0.05645332, 0.04465462],
                   [-0.88535499, -0.4638495, -0.03146717, 0.04170271],
                   [-0.00161712, -0.06461072, 0.99790923, 0.156906],
                   [0., 0., 0., 1.]])
x0 = f.rvec6__t_4_4(x0_mat)
ls = f.least_squares(residual, x0, loss="cauchy", jac="3-point")

# print(num_data)
print(ls.x)

np.save("c2e", f.inv(f.t_4_4__rvec6(ls.x)))

print(ls.cost)
print(ls.optimality)

