import numpy as np
import matplotlib.pyplot as plt
from newton import Newton
from gradient_descent import GD
from function import Function


def run(step, point_type, lambda_p, method):

    f = Function(lambda_p)

    n = f.get_size()  # x size

    # Initial point
    if point_type == "const":
        x0 = np.ones(n)
    else:
        x0 = np.random.uniform(-2, 2, n)

    mxitr = 50000  # Max number of iterations

    tol_g = 1e-8  # Tolerance for gradient

    tol_x = 1e-8  # Tolerance for x

    tol_f = 1e-8  # Tolerance for function

    # Method for step update
    if step == "fijo":
        msg = "StepFijo"

    elif step == "hess":
        msg = "StepHess"

    else:
        msg = "Backtracking"

    step_size = 1  # Gradient step size for "StepFijo" method

    # Estimate minimum point through optimization method chosen
    if method == "gd":
        alg = GD()
    elif method == "newton":
        alg = Newton()
    else:
        print("\n Error: Invalid optimization method: %s\n" % method)
        return

    xs = alg.iterate(x0, mxitr, tol_g, tol_x, tol_f, f, msg, step_size)

    # Print point x found and function value f(x)
    # print("\nPoint x found: ", xs[-1])
    print("\nf(x) =  ", f.eval(xs[-1]))

    plt.plot(np.array(range(n)), xs[-1])
    plt.plot(np.array(range(n)), f.y)
    plt.legend(['x*', 'y'], loc = 'best')
    plt.show()
