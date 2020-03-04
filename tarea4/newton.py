import numpy as np
from numpy.linalg import inv
from scipy.linalg import ldl

class Newton:
    """Newton method for approximating local minimum point x*
    for a given function f starting at point x0.

    '''

    Returns
    -------
    List
        a list with all points x traversed during gradient descent
    """

    def iterate(self, x0, mxitr, tol_g, tol_x, tol_f, f, msg, *args):
        """Iterate over x_{k+1} = x_k - alpha_k * d_k

        Where d_k = Hessian_inverse * gradient is the direction of descent.

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

            hess = f.hessian(x)  # Get hessian of function evaluated at x

            # Calculate step size depending on value of msg
            if msg == "StepFijo":

                try:
                    alpha = args[0]
                except ValueError as err:
                    print("\n Step size value not given: ", err)

            elif msg == "StepHess":
                alpha = (grad.T.dot(grad)) / (grad.dot(hess).dot(grad.T))

            elif msg == "Backtracking":
                alpha = self.backtracking(x, grad, f, 0.01, 0.5)

            else:
                print("\n Invalid step size update method\n")
                break

            # Make sure hessian is positive definite
            hess = self.cholesky_identity(hess)

            # Update x value
            x_old = x
            d_k = inv(hess).dot(grad)
            x = x - alpha * d_k

            # Calculate different tolerance criteria
            tol_x_val = np.linalg.norm(x - x_old) / max(1.0, np.linalg.norm(x_old))
            tol_f_val = np.absolute(f.eval(x) - f.eval(x_old)) / max(1.0, np.absolute(f.eval(x_old)))
            tol_g_val = np.linalg.norm(x_old)

            if k%1 == 0:
                self.log2(x_old, grad, x, k, tol_g_val, np.linalg.norm(x - x_old), f.eval(x))
                # self.log(x_old, grad, x, k, tol_g_val, np.linalg.norm(x - x_old), f(x))

            k += 1  # Update iteration counter

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

    def cholesky_identity(self, H, beta = 1e-3):
        """Routine to make matrix H positive definite by adding tau to the
        diagonal until cholesky factorization works without raising
        any error, i.e., find H_pd = H + tau_k*I such that H_pd is a
        positive definite matrix.

        Args:
            H : numpy.array
                Square matrix to modify until it´s positive definite
            beta : float
                Optional parameter for choosing

        Returns
        -------
        numpy.array
            Positive definite matrix H_pd = H + tau_k*I
        """

        if np.min(H.diagonal()) > 0:
            tau = 0
        else:
            tau = -np.min(H.diagonal()) + beta

        for i in range(100):
            # print("try cholesky %s" % i)
            try:
                ldl(H + tau*np.identity(H.shape[0]))
                break

            except ValueError:
                tau = max(2*tau, beta)

        return H + tau*np.identity(H.shape[0])

    def cholesky(self, H):
        """Performs a Cholesky decomposition of H, which must
        be a symmetric and positive definite matrix.

        Args:
            H : numpy.array
                Square matrix assumed to be symmetric and positive definie

        Returns
        -------
        numpy.array
            cholesky decomposition matrix of H
        """

        n = H.shape[0]

        L = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1):

                sum = np.sum(L[i][k] * L[j][k] for k in range(j))

                if i == j:  # Diagonal entries
                    if H[j][j] - sum < 0:
                        raise Exception('ValueError')
                    L[j][j] = np.sqrt(H[j][j] - sum)

                else:
                    L[i][j] = (H[i][j] - sum)/L[j][j]

        return L

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
