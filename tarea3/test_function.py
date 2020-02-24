import numpy as np
from newton import Newton
from function import Function


def run(step, point_type, lambda_p):

    f = Function(lambda_p)

    # Initial point
    if point_type == "const":
        x0 = np.ones(f.get_size())
    else:
        x0 = np.random.uniform(-2, 2, f.get_size())

    mxitr = 10000  # Max number of iterations

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

    step_size = 1e-5  # Gradient step size for "StepFijo" method

    # Estimate minimum point through Newton method
    newton_alg = Newton()
    xs = newton_alg.iterate(x0, mxitr, tol_g, tol_x, tol_f, f, msg, step_size)

    # Print point x found and function value f(x)
    # print("\nPoint x found: ", xs[-1])
    print("\nf(x) =  ", f.eval(xs[-1]))
