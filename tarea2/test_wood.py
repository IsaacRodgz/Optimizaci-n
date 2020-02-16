import numpy as np
from gradient_descent import gradient_descent


# Wood function

def f_wood(x):

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    return sum((
        100*(x1**2 - x2)**2,
        (x1-1)**2,
        (x3-1)**2,
        90*(x3**2 - x4)**2,
        10.1*((x2-1)**2 + (x4-1)**2),
        19.8*(x2-1)*(x4-1),
        ))


# Gradient of Wood function

def g_wood(x):

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    return np.array([
            400*x1*(x1**2-x2) + 2*(x1-1),
            -200*(x1**2-x2) + 20.2*(x2-1) + 19.8*(x4-1),
            2*(x3-1) + 360*x3*(x3**2-x4),
            -180*(x3**2-x4) + 20.2*(x4-1) + 19.8*(x2-1)
            ])


# Hessian of Wood function

def H_wood(x):

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    hessian = np.zeros((4, 4))

    hessian[0][0] = 1200*x1**2 - 400*x2 + 2
    hessian[0][1] = -400*x1

    hessian[1][0] = -400*x1
    hessian[1][1] = 220.2
    hessian[1][3] = 19.8

    hessian[2][2] = 1080*x3**2 - 360*x4 + 2
    hessian[2][3] = -360*x3

    hessian[3][1] = 19.8
    hessian[3][2] = -360*x3
    hessian[3][3] = 200.2

    return hessian


if __name__ == '__main__':

    # Initial point
    x0 = np.array([-3, -1, -3, -1])

    # Min point
    x_min = np.array([1, 1, 1, 1])

    # Max number of iterations
    mxitr = 10000

    # Tolerance for gradient
    tol_g = 1e-8

    # Tolerance for x
    tol_x = 1e-8

    # Tolerance for function
    tol_f = 1e-8

    # Method for step update
    # msg = "StepFijo"
    # msg = "StepHess"
    # msg = "Backtracking"

    # Gradient step size for "StepFijo" method
    step_size = 2e-4

    # Estimate minimum point through gradient descent
    xs = gradient_descent(x0, mxitr, tol_g, tol_x, tol_f,
                          f_wood, g_wood, msg, H_wood,
                          step_size)

    # Print point x found and function value f(x)

    print("\nPoint x found: ", xs[-1])
    print("\nf(x) =  ", f_wood(xs[-1]))
