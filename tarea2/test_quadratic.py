import numpy as np
from gradient_descent import gradient_descent
from plot import plot_level_set


def f(x):

    return (x[0]-3.8)**2 + (x[1] - 4.2)**2


def g(x):

    return np.array([2*(x[0]-3.8), 2*(x[1]-4.2)])


def H(x):

    return np.array([[2, 0], [0, 2]])


if __name__ == '__main__':

    # Initial point
    x0 = np.array([-1.2, 1])

    # Max number of iterations
    mxitr = 10000

    # Tolerance for gradient
    tol_g = 1e-8

    # Tolerance for x
    tol_x = 1e-8

    # Tolerance for function
    tol_f = 1e-8

    # Method for step update
    msg = "StepFijo"
    # msg = "StepHess"
    # msg = "Backtracking"

    # Gradient step size for "StepFijo" method
    step_size = 2e-3

    # Estimate minimum point through gradient descent
    xs = gradient_descent(x0, mxitr, tol_g, tol_x, tol_f,
                          f, g, msg, H, step_size)

    # Plot level sets and gradient path

    plot_level_set(xs, f, -2.0, 6.0, -8.0, 8.0, x0, np.array([3.8, 4.2]))
