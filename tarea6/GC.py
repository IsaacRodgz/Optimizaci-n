import numpy as np


class GC:
    """Conjugate Gradient method for calculating minimum point of quadratic
    function starting at point x0.

    '''

    Returns
    -------
    List
        a list with all points x traversed during conjugate gradient
    List
        a list with values f(x) traversed during conjugate gradient
    """

    def iterate(self, x0, f):
        """Iterate with Conjugate Gradient algorithm

        Parameters
        ----------
        x0 : numpy array
            Initial point
        f : Function object
            Objective function with eval, gradient and hessian methods

        Returns
        -------
        List
            a list with all points x traversed during conjugate gradient
        List
            a list with values f(x) traversed during conjugate gradient
        """

        x = x0  # Start with initial point

        xs = []  # List to save points x

        fs = []  # List with all the values f(x) traversed

        grad = f.gradient(x0)  # Get initial gradient evaluated at point x0

        d = -grad  # Get initial conjugate direction

        Q = f.get_Q()  # Get matrix of quadratic function

        # Iterate at most the dimension of the problem
        for k in range(x.shape[0]):

            xs.append(x)
            fs.append(f.eval(x))

            alpha = -(grad.dot(d))/(d.dot(Q).dot(d))
            x = x + alpha*d
            g = f.gradient(x)
            beta = (g.dot(Q).dot(d))/(d.dot(Q).dot(d))
            d = -g + beta*d

            if k%20 == 0:
                print("Iter {0}: f(x) = {1}".format(k, fs[-1]))

        xs.append(x)
        fs.append(f.eval(x))

        return xs, fs

    def log(self, x_old, grad, x, curr_iter, tol_g_val, tol_x_val, tol_f_val):
        """Print to console status of current iteration

        Args:
            x_old : numpy.array
                Previous solution point
            grad : numpy.array
                Gradient of the function at x_old
            x : numpy.array
                Solution point after gradient step
            curr_iter : int
                Current number of iteration
            tol_g : float
                Tolerance for gradient norm
            tol_x : float
                Tolerance for x's relative error
            tol_f : float
                Tolerance for relative error in evaluation of function f

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
            x_old : numpy.array
                Previous solution point
            grad : numpy.array
                Gradient of the function at x_old
            x : numpy.array
                Solution point after gradient step
            curr_iter : int
                Current number of iteration
            tol_g : float
                Tolerance for gradient norm
            tol_x : float
                Tolerance for x's relative error
            tol_f : float
                Tolerance for relative error in evaluation of function f

        Output: Print to console status of gradient descent
        """
        # print("-----------------------------------")
        # print("\n  k  |  ||x_k+1 - x_k||  |  || grad(f_k) ||  |  f(x_k)")
        print("{0} & {1:.10E} & {2:.10E} & {3:.10E} \\\\".format(curr_iter, tol_x_val, tol_g_val, tol_f_val))
