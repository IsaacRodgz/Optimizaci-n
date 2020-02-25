import numpy as np


class GD:
    """Gradient method for approximating local minimum point x*
    for a given function f starting at point x0.

    '''

    Returns
    -------
    List
        a list with all points x traversed during gradient descent
    """

    def iterate(self, x0, mxitr, tol_g, tol_x, tol_f, f, msg, *args):
        """Iterate over x_{k+1} = x_k - alpha_k * d_k

        Where d_k = gradient(f(x_k))

        Parameters
        ----------
        x0 : numpy array
            Initial point
        mxitr : int
            Max. number of iterations
        tol_g : float
            Tolerance for gradient norm
        tol_x : float
            Tolerance for x's relative error
        tol_f : float
            Tolerance for relative error in evaluation of function f
        f : Function object
            Objective function with eval, gradient and hessian methods
        msg : string
            Step size update method

        Returns
        -------
        List
            a list with all points x traversed during gradient descent
        """

        k = 0  # Start iteration at 0

        x = x0  # Start with initial point

        xs = []  # List to save points x

        # Iterate while max. num. of iters has not been reached
        while k < mxitr:

            xs.append(x)  # Save current point

            grad = f.gradient(x)  # Get gradient evaluated at point x

            # Calculate step size depending on value of msg
            if msg == "StepFijo":

                try:
                    alpha = args[0]
                except ValueError as err:
                    print("\n Step size value not given: ", err)

            elif msg == "StepHess":
                hess = f.hessian(x)  # Get hessian of function evaluated at x
                alpha = (grad.T.dot(grad)) / (grad.dot(hess).dot(grad.T))

            elif msg == "Backtracking":
                alpha = self.backtracking(x, grad, f, 0.01, 0.5)

            else:

                print("\n Invalid step size update method\n")
                break

            # Update x value
            x_old = x
            x = x - alpha * grad

            # Calculate different tolerance criteria
            tol_x_val = np.linalg.norm(x - x_old) / max(1.0, np.linalg.norm(x_old))
            tol_f_val = np.absolute(f.eval(x) - f.eval(x_old)) / max(1.0, np.absolute(f.eval(x_old)))
            tol_g_val = np.linalg.norm(x_old)

            if k%1 == 0:
                self.log2(x_old, grad, x, k, tol_g_val, np.linalg.norm(x - x_old), f.eval(x))

            #log(x_old, grad, x, k, tol_g_val, np.linalg.norm(x - x_old), f(x))

            # Update iteration counter

            k += 1

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

        #self.log2(x_old, grad, x, k, tol_g_val, np.linalg.norm(x - x_old), f.eval(x))

        return xs

    def backtracking(self, x, grad, f, tau, beta):
        """Calculate step size through backtracking

        Args:
            x : numpy.array
                Current iteration point
            grad : numpy.array
                Gradient of function f at point x
            f : Function object
                Function to minimize
            tau : float
                Algorithm parameter
            beta : float
                Algorithm parameter

        Returns
        -------
        float
            step size found by backtracking
        """

        alpha = 1

        while f.eval(x - alpha*grad) > f.eval(x) + tau*alpha*f.gradient(x).dot(grad):
            alpha *= beta

        return alpha

    def log(self, x_old, grad, x, curr_iter, tol_g_val, tol_x_val, tol_f_val):
        """Print to console status of current iteration

        Args:
            x_old (numpy array): Previous solution point
            grad (numpy array): Gradient of the function at x_old
            x (numpy array): Solution point after gradient step
            curr_iter (int): Current number of iteration
            tol_g (float): Tolerance for gradient norm
            tol_x (float): Tolerance for x's relative error
            tol_f (float): Tolerance for relative error in evaluation of function f

        Output: Print to console status of gradient descent
        """
        print("-----------------------------------")
        print("\n Iter: ", curr_iter)
        print("\n x_old: ", x_old)
        print("\n gradient: ", grad)
        print("\n x: ", x)
        print("\n tol_x_val: ", tol_x_val)
        print("\n tol_f_val: ", tol_f_val)
        print("\n tol_g_val: %s \n " % tol_g_val)

    def log2(self, x_old, grad, x, curr_iter, tol_g_val, tol_x_val, tol_f_val):
        """Print to console status of current iteration

        Args:
            x_old (numpy array): Previous solution point
            grad (numpy array): Gradient of the function at x_old
            x (numpy array): Solution point after gradient step
            curr_iter (int): Current number of iteration
            tol_g (float): Tolerance for gradient norm
            tol_x (float): Tolerance for x's relative error
            tol_f (float): Tolerance for relative error in evaluation of function f

        Output: Print to console status of gradient descent
        """
        #print("-----------------------------------")
        # print("\n  k  |  ||x_k+1 - x_k||  |  || grad(f_k) ||  |  f(x_k)")
        print("{0} & {1:.10E} & {2:.10E} & {3:.10E} \\\\".format(curr_iter, tol_x_val, tol_g_val, tol_f_val))
