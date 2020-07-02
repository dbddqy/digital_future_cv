import libs.lib_optimization as opt
import numpy as np
import cv2

# def f(x):
#     return (x-5)*(x-5)*(x-5)*(x-5)
#
#
# def f_d(x):
#     return 4*(x-5)*(x-5)*(x-5)
#
#
# def f_d_d(x):
#     return 12*(x-5)*(x-5)
#
#
# x0 = 100
#
# for _ in range(20):
#     x0 = x0 - (f_d(x0) / f_d_d(x0))
#     print("x: %f" % x0)
#     print("f(x): %f" % f(x0))


# w2k = []
# w2k.append(opt.rvec6__t_4_4(np.eye(4)))
# w2k.append(np.array([-5.48993849e-03, -1.44357842e-02, -1.59200111e+00, -8.94487825e-03, -4.21923127e-01, 1.56634777e-03]))
# w2k.append(np.array([1.39512364e-02, 2.54124592e-02, -1.56722684e+00, 1.34946692e-03, 3.87557431e-01, 4.51886330e-03]))
# print(w2k)
# np.save("w2k", w2k)
#
# # print(opt.t_4_4__rvec6(np.array([0.16259024, -0.14102478, -2.39414456, 0.17434413, 0.04828597, 0.08690468])))
# print(opt.t_4_4__rvec6(np.array([-0.03847483, 0.06741132, -2.05344637, 0.04465462, 0.04170271, 0.156906])))
#
# t = np.array([0.0510735, 0.0414825, 0.155161]).reshape([3, 1])
# R = np.array([-0.465712, -0.884916, -0.0059336, 0.883495, -0.46456, -0.0601688, 0.0504879, -0.0332636, 0.998171]).reshape([3, 3])
# t_4_4 = np.eye(4)
# t_4_4[0:3, 0:3] = R.T
# t_4_4[0:3, 3:4] = t
# np.save("c2e", opt.inv(t_4_4))


rec = np.loadtxt("w2k.txt")
print(rec)
np.save("w2k", rec)

# img = cv2.imread("..\\data_2D\\20200701_marker5\marker_6.png")
# print(img)

