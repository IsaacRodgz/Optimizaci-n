import numpy as np
import matplotlib.pyplot as plt
from dogleg import Dogleg
from Rosenbrock import Rosenbrock


def run(step, point_type):

    n = 2
    f = Rosenbrock(n)

    # Initial point
    if point_type == "const":
        x0 = np.ones(n)
        x0[0] = -1.2
        #x0[-2] = -1.2
    else:
        x0 = np.random.uniform(-2, 2, n)

    mxitr = 50000  # Max number of iterations
    tol_g = 1e-8  # Tolerance for gradient
    tol_x = 1e-8  # Tolerance for x
    tol_f = 1e-8  # Tolerance for function

    alg = Dogleg()
    xs = alg.iterate(x0, mxitr, tol_g, tol_x, tol_f, f)

    plt.plot(np.array(range(n)), xs[-1])
    plt.legend(['x*'], loc = 'best')
    plt.show()
