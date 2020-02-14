import numpy as np
from gradient_descent import gradient_descent


def f(x):

    return (x[0]-3.8)**2 + (x[1] - 4.2)**2


def g(x):

    return np.array([2*(x[0]-3.8), 2*(x[1]-4.2)])


def H(x):

    return np.array([[2, 0], [0, 2]])


if __name__ == '__main__':

    x0 = np.array([0.1, 0.1])
    mxitr = 400
    tol_g = 1e-3
    tol_x = 1e-6
    tol_f = 1e-6
    msg = "StepFijo"
    step_size = 1e-2

    gradient_descent(x0, mxitr, tol_g, tol_x, tol_f, f, g, msg, H, step_size)
