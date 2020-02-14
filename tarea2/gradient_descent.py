import numpy as np


def gradient_descent(x0, mxitr, tol_g, tol_x, tol_f, f, g, msg, H=None, *args):
    """Approximate local minimum point x* for a given function f starting at point x0

    Args:
        x0 (numpy array): Initial point
        mxitr (int): Max. number of iterations
        tol_g (float): Tolerance for gradient norm
        tol_x (float): Tolerance for x's relative error
        tol_f (float): Tolerance for relative error in evaluation of function f
        f (function): Objective function
        g (function): Gradient function
        msg (string): Step size update method
        H (function): Hessian  function (Optional parameter)

    Returns: Numpy array with minimum point
    """
    # Start iteration at 0
    k = 0

    # Initial point
    x = x0

    # Iterate while max. num. of iters has not been reached
    while k < mxitr:

        # Get gradient evaluated at point x
        grad = g(x)

        # Calculate step size depending on value of msg
        if msg == "StepFijo":

            try:
                alpha = args[0]

            except ValueError as err:
                print("\n Step size value not given: ", err)

        elif msg == "StepHess":

            alpha = (grad.T.dot(grad))/(grad.dot(H(x)).dot(grad.T))

        elif msg == "Backtracking":

            alpha = 0.01

        else:

            print("\n Invalid step size update method\n")
            break

        # Update x value

        x_old = x

        x = x - alpha * grad

        # Calculate different tolerance criteria

        k += 1

        tol_x_val = np.linalg.norm(x - x_old)/max(1.0, np.linalg.norm(x_old))

        tol_f_val = np.absolute(f(x) - f(x_old))/max(1.0, np.absolute(f(x_old)))

        tol_g_val = np.linalg.norm(x_old)

        log(x_old, x, k, tol_g_val, tol_x_val, tol_f_val)

        # Check for convergence

        if tol_x_val < tol_x:

            print("\n Algorithm converged in x\n")

            break

        if tol_f_val < tol_f:

            print("\n Algorithm converged in f\n")

            break

        if tol_g_val < tol_g:

            print("\n Algorithm converged in g\n")

            break

        if k > mxitr:

            print("\n Algorithm reached max num of iterations\n")

            break

    return x


def log(x_old, x, curr_iter, tol_g_val, tol_x_val, tol_f_val):
    """Print to console status of current iteration

    Args:
        x_old (numpy array): Previous solution point
        x (numpy array): Solution point after gradient step
        curr_iter (int): Current number of iteration
        tol_g (float): Tolerance for gradient norm
        tol_x (float): Tolerance for x's relative error
        tol_f (float): Tolerance for relative error in evaluation of function f

    Output: Print to console status of gradient descent
    """
    print("-----------------------------------")
    print("\n Iter: ", curr_iter+1)
    print("\n x_old: ", x_old)
    print("\n x: ", x)
    print("\n tol_x_val: ", tol_x_val)
    print("\n tol_f_val: ", tol_f_val)
    print("\n tol_g_val: %s \n " % tol_g_val)
