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
w2k.append(np.array([1.56146028e-02, 1.08915129e-02, -3.12339231e+00, 3.13623572e-01, 1.13036116e-03, 9.70693834e-04]))
print(w2k)
np.save("w2k", w2k)
