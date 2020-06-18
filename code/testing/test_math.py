import libs.lib_optimization as opt
import numpy as np

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


w2k = []
w2k.append(opt.rvec6__t_4_4(np.eye(4)))
w2k.append(np.array([1.77132991e-02, 7.39518859e-03, -3.12274377e+00, 3.13270281e-01, 1.11441898e-03, 1.10271995e-03]))
print(w2k)
np.save("w2k", w2k)
