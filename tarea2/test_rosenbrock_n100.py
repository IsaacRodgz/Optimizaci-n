import numpy as np
from gradient_descent import gradient_descent


# Rosenbrock function (n = 100)

def f_rosenbrock_100(x):

    sum = 0

    for i in range(99):

        sum += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2

    return sum


# Gradient of Rosenbrock function

def g_rosenbrock_100(x):

    grad = []

    for i in range(99):

        grad.append(-400*(x[i]*x[i+1] - x[i]**3) + 2*(x[i] - 1))

    grad.append(200*(x[99] - x[98]**2))

    return np.array(grad)


# Hessian of Rosenbrock function

def H_rosenbrock_100(x):

    hessian = np.zeros((100, 100))

    for i in range(99):

        hessian[i][i] = -400*x[i+1] + 1200*x[i]**2 + 2
        hessian[i][i+1] = -400*x[i]

    hessian[99][98] = -400*x[98]
    hessian[99][99] = 200

    return hessian


if __name__ == '__main__':

    # Initial point
    x0 = np.ones(100)
    x0[0] = -1.2
    x0[-2] = -1.2

    # Min point
    x_min = np.ones(100)

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
                          f_rosenbrock_100, g_rosenbrock_100, msg,
                          H_rosenbrock_100, step_size)

    # Print point x found and function value f(x)

    print("\nPoint x found: ", xs[-1])
    print("\nf(x) =  ", f_rosenbrock_100(xs[-1]))
