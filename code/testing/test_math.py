def f(x):
    return (x-5)*(x-5)*(x-5)*(x-5)


def f_d(x):
    return 4*(x-5)*(x-5)*(x-5)


def f_d_d(x):
    return 12*(x-5)*(x-5)


x0 = 100

for _ in range(20):
    x0 = x0 - (f_d(x0) / f_d_d(x0))
    print("x: %f" % x0)
    print("f(x): %f" % f(x0))
