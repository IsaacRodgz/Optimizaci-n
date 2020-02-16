import numpy as np
from gradient_descent import gradient_descent
from plot import plot_level_set


# Rosenbrock function (n = 2)

def f_rosenbrock_2(x):

    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


# Gradient of Rosenbrock function

def g_rosenbrock_2(x):

    return np.array([-400*(x[0]*x[1] - x[0]**3) + 2*(x[0] - 1), 200*(x[1] - x[0]**2)])


# Hessian of Rosenbrock function

def H_rosenbrock_2(x):

    return np.array([[-400*x[1] + 1200*x[0]**2 + 2, -400*x[0]], [-400*x[0], 200]])


if __name__ == '__main__':

    # Initial point
    x0 = np.array([-1.2, 1])

    # Min point
    x_min = np.array([1, 1])

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
                          f_rosenbrock_2, g_rosenbrock_2, msg, H_rosenbrock_2,
                          step_size)

    # Print point x found and function value f(x)

    print("\nPoint x found: ", xs[-1])
    print("\nf(x) =  ", f_rosenbrock_2(xs[-1]))

    # Plot level sets and gradient path

    plot_level_set(xs, f_rosenbrock_2, -5.0, 2.0, -8.0, 8.0, x0, x_min)
