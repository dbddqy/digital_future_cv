# import scipy as sp
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

import numpy as np


def inv(t_4_4):
    t_4_4_new = t_4_4.copy()
    t_4_4_new[0:3, 3:4] = -t_4_4[0:3, 0:3].T.dot(t_4_4[0:3, 3:4])
    t_4_4_new[0:3, 0:3] = t_4_4[0:3, 0:3].T
    return t_4_4_new


def t_4_4__quat(quat, t):
    t_4_4 = np.eye(4)
    t_4_4[0:3, 0:3] = R.from_quat(quat).as_matrix()
    t_4_4[0:3, 3:4] = t.reshape([3, 1])
    return t_4_4


def t_4_4__rvec(rvec, t):
    t_4_4 = np.eye(4)
    t_4_4[0:3, 0:3] = R.from_rotvec(rvec).as_matrix()
    t_4_4[0:3, 3:4] = t.reshape([3, 1])
    return t_4_4


def rvec__t_4_4(t_4_4):
    vec = np.zeros([6, ])
    vec[0:3] = R.from_matrix(t_4_4[0:3, 0:3]).as_rotvec()
    vec[3:6] = t_4_4[0:3, 3:4].reshape([3, ])
    return vec
