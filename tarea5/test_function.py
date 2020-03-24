import numpy as np
import matplotlib.pyplot as plt
from dogleg import Dogleg
from lstr import LSTR
from Rosenbrock import Rosenbrock


def run(method, point_type):

    n = 100
    f = Rosenbrock(n)

    # Initial point
    if point_type == "const":
        x0 = np.ones(n)
        x0[0] = -1.2
        x0[-2] = -1.2
    else:
        x0 = np.random.uniform(-2, 2, n)

    mxitr = 5000  # Max number of iterations
    tol_g = 1e-8  # Tolerance for gradient
    tol_x = 1e-8  # Tolerance for x
    tol_f = 1e-8  # Tolerance for function

    if method == "dogleg":
        alg = Dogleg()
    elif method == "lstr":
        alg = LSTR()
    else:
        print("\n Error. {} is not a valid method".format(method))
        return

    xs, vs = alg.iterate(x0, mxitr, tol_g, tol_x, tol_f, f, "3")

    plt.plot(np.array(range(len(vs))), vs)
    plt.legend(['f(x)'], loc = 'best')
    plt.show()
