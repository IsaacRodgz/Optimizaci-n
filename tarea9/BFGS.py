import numpy as np
from scipy.optimize import line_search


class BFGS:

    def iterate(self, x0, f, grad, mx_iter, grad_tol):
        """Iterate with BFGS algorithm

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

        xs = []  # List to save points x

        fs = []  # List with all the values f(x) traversed

        gs = []  # List with all the values ||gradient(x)|| traversed

        x = x0  # Start with initial point

        H = np.eye(x0.shape[0])  # Initial approximation of Hessian

        g = grad(x0)  # Get initial gradient evaluated at point x0

        # Iterate at most the dimension of the problem
        k = 0
        while k < mx_iter and np.linalg.norm(g) > grad_tol:

            # Get direction
            d = -np.dot(H, g)

            # Estimate alpha through line search method
            alpha, _, _, _, _, _ = line_search(f, grad, x, d, g, None, None, c1 = 1e-4, c2 = 0.4)

            if alpha is None:
                alpha = self.cubic_interpolation(x, -g, f, 1e-4)

            # Calculate new x
            x = x + alpha*d

            # Update gradient
            g_new = grad(x)
            y = g_new - g
            s = alpha*d
            g = g_new

            if np.fabs(np.dot(s,y)) <= 1e-8:
                rho = 1
            else:
                rho = 1/np.dot(s,y)

            sy = np.outer(s,y)
            left = (np.eye(H.shape[0]) - rho*sy)
            right = (np.eye(H.shape[0]) - rho*sy.T)
            H = left@H@right + rho*np.outer(s,s)

            xs.append(x)
            fs.append(abs(f(x)))
            gs.append(np.linalg.norm(g))

            if k%1 == 0:
                print("Iter {0}: f(x) = {1}    |g(x)| = {2}".format(k, fs[-1], gs[-1]))

            k += 1

        return xs, fs, gs

    def backtracking(self, x, grad, d, f, tau, beta):
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

        while f.eval(x - alpha*d) > f.eval(x) + tau*alpha*grad.dot(d):
            alpha *= beta

        return alpha

    def cubic_interpolation(self, x, d, f, alpha_init):
        """Calculate step size through cubic interpolation

        Args:
            x : numpy.array
                Current iteration point
            d : numpy.array
                Current direction of descent (-grad(x))
            f : Function object
                Function to minimize
            alpha_init : float
                Initial step size

        Returns
        -------
        float
            step size found by cubic interpolation
        """

        c_1 = 1e-4

        alpha_0 = alpha_init

        phi_p0 = -d.dot(d)
        phi_0 = f(x)
        phi_alpha_0 = f(x+alpha_0*d)

        # Check if alpha_init satisfies armijo conditions

        if phi_alpha_0 <= phi_0 + c_1*alpha_0*(phi_p0):
            return alpha_0

        alpha_1 = (-(alpha_0**2)*phi_p0) / (2*(phi_alpha_0 - phi_p0*alpha_0 - phi_0))

        # Check if alpha_1, obtained by cuadratic interpolation, satisfies armijo conditions

        phi_alpha_1 = f(x+alpha_1*d)

        if phi_alpha_1 > phi_0 + c_1*alpha_1*(phi_p0):
            return alpha_1

        # Perform cubic interpolation

        constant = 1/(alpha_0**2*alpha_1**2*(alpha_1-alpha_0))
        m1 = np.array([[alpha_0**2, -alpha_1**2], [-alpha_0**3, alpha_1**3]])
        m2 = np.array([phi_alpha_1-phi_p0*alpha_1-phi_0, phi_alpha_0-phi_p0*alpha_0-phi_0])

        a, b = constant*m1.dot(m2)
        c = phi_p0

        alpha_2 = (-b + np.sqrt(b**2 - 3*a*c)) / (3*a)

        while f(x+alpha_2*d) > phi_0 + c_1*alpha_2*(phi_p0):
            alpha_0 = alpha_1
            alpha_1 = alpha_2

            constant = 1/(alpha_0**2*alpha_1**2*(alpha_1-alpha_0))
            m1 = np.array([[alpha_0**2, -alpha_1**2], [-alpha_0**3, alpha_1**3]])
            m2 = np.array([phi_alpha_1-phi_p0*alpha_1-phi_0, phi_alpha_0-phi_p0*alpha_0-phi_0])

            a, b = constant*m1.dot(m2)

            alpha_2 = (-b + np.sqrt(b**2 - 3*a*c)) / (3*a)

            return alpha_2

        return alpha_2

    def barzilai_borwein(self, x_all, f):
        """Calculate step size through Barzilai-Borwein

        Args:
            x_all : list
                List of all points x iterated until now
            f : Function object
                Function to minimize

        Returns
        -------
        float
            step size found by Barzilai-Borwein
        """

        s_k = x_all[-1] - x_all[-2]
        y_k = f.gradient(x_all[-1]) - f.gradient(x_all[-2])

        return s_k.dot(s_k)/(s_k.dot(y_k))

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
