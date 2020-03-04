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

    def iterate(self, x0, mxitr, tol_g, tol_x, tol_f, f, msg, function, *args):
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

        alpha = 1e-3 # Initial step size

        if msg == "zhang":
            return self.zhang_hager(x0, f, tol_f, mxitr)

        # Iterate while max. num. of iters has not been reached
        while k < mxitr:

            xs.append(x)  # Save current point

            grad = f.gradient(x)  # Get gradient evaluated at point x

            # Calculate step size depending on value of msg
            if msg == "barzilai":
                if k == 0:
                    pass
                else:
                    alpha = self.barzilai_borwein(xs, f)

            elif msg == "cubic":
                alpha = self.cubic_interpolation(x, -grad, f, alpha)

            else:
                print("\n Invalid step size update method\n")
                break

            # Update x value
            x_old = x
            x = x - alpha * grad

            if function == "mnist":
                error = f.error(x)
                loss = f.eval(x)

                self.log(k, error, loss)

                if error < tol_f:
                    print("\n Algorithm converged in error\n")
                    break

            else:
                # Calculate different tolerance criteria
                tol_x_val = np.linalg.norm(x - x_old) / max(1.0, np.linalg.norm(x_old))
                tol_f_val = np.absolute(f.eval(x) - f.eval(x_old)) / max(1.0, np.absolute(f.eval(x_old)))
                tol_g_val = np.linalg.norm(x_old)

                if k%1 == 0:
                    self.log2(x_old, grad, x, k, tol_g_val, tol_x_val, f.eval(x))

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

            # Update iteration counter
            k += 1

        return xs

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
        phi_0 = f.eval(x)
        phi_alpha_0 = f.eval(x+alpha_0*d)

        # Check if alpha_init satisfies armijo conditions

        if phi_alpha_0 <= phi_0 + c_1*alpha_0*(phi_p0):
            return alpha_0

        alpha_1 = (-(alpha_0**2)*phi_p0) / (2*(phi_alpha_0 - phi_p0*alpha_0 - phi_0))

        # Check if alpha_1, obtained by cuadratic interpolation, satisfies armijo conditions

        phi_alpha_1 = f.eval(x+alpha_1*d)

        if phi_alpha_1 > phi_0 + c_1*alpha_1*(phi_p0):
            return alpha_1

        # Perform cubic interpolation

        constant = 1/(alpha_0**2*alpha_1**2*(alpha_1-alpha_0))
        m1 = np.array([[alpha_0**2, -alpha_1**2], [-alpha_0**3, alpha_1**3]])
        m2 = np.array([phi_alpha_1-phi_p0*alpha_1-phi_0, phi_alpha_0-phi_p0*alpha_0-phi_0])

        a, b = constant*m1.dot(m2)
        c = phi_p0

        alpha_2 = (-b + np.sqrt(b**2 - 3*a*c)) / (3*a)

        while f.eval(x+alpha_2*d) > phi_0 + c_1*alpha_2*(phi_p0):
            alpha_0 = alpha_1
            alpha_1 = alpha_2

            constant = 1/(alpha_0**2*alpha_1**2*(alpha_1-alpha_0))
            m1 = np.array([[alpha_0**2, -alpha_1**2], [-alpha_0**3, alpha_1**3]])
            m2 = np.array([phi_alpha_1-phi_p0*alpha_1-phi_0, phi_alpha_0-phi_p0*alpha_0-phi_0])

            a, b = constant*m1.dot(m2)

            alpha_2 = (-b + np.sqrt(b**2 - 3*a*c)) / (3*a)

            return alpha_2

        return alpha_2

    def cuadratic_interpolation(self, x, d, f, alpha_init, c):
        """Calculate step size through cuadratic interpolation

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
            step size found by cuadratic interpolation
        """

        c_1 = 1e-4
        alpha_0 = alpha_init

        phi_p0 = -d.dot(d)
        phi_0 = f.eval(x)
        phi_alpha_0 = f.eval(x+alpha_0*d)

        alpha_1 = (-(alpha_0**2)*phi_p0) / (2*(phi_alpha_0 - phi_p0*alpha_0 - phi_0))

        while f.eval(x+alpha_1*d) > c + c_1*alpha_1*(phi_p0):
            alpha_0 = alpha_1
            phi_alpha_0 = f.eval(x+alpha_0*d)
            alpha_1 = (-(alpha_0**2)*phi_p0) / (2*(phi_alpha_0 - phi_p0*alpha_0 - phi_0))

        return alpha_1

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

    def zhang_hager(self, x0, f, tol_f, mxitr):
        """Iterate over x_{k+1} = x_k - alpha_k * d_k through Zhang-Hager
        Where d_k = gradient(f(x_k))

        Args:
            x : numpy.array
                Current iteration point
            f : Function object
                Function to minimize
            tol_f : float
                Tolerance for relative error in evaluation of function f

        Returns
        -------
        List
            a list with all points x traversed during Zhang-Hager descent
        """

        eta = 0.1
        c = f.eval(x0)
        q = 1
        x = x0
        xs = []
        k = 0

        while(np.linalg.norm(f.gradient(x)) > tol_f and k < mxitr):

            xs.append(x) #  Save current point
            grad = f.gradient(x)  # Get gradient evaluated at point x

            # Find alpha through cuadratic interpolation
            alpha = self.cuadratic_interpolation(x, -grad, f, 0.1, c)

            # Update solution
            x_old = x
            x = x - alpha * grad

            # Update parameters
            q_old = q
            q = eta*q + 1
            c = (eta*q_old*c + f.eval(x))/q

            # Calculate different tolerance criteria
            tol_x_val = np.linalg.norm(x - x_old) / max(1.0, np.linalg.norm(x_old))
            tol_f_val = np.absolute(f.eval(x) - f.eval(x_old)) / max(1.0, np.absolute(f.eval(x_old)))
            tol_g_val = np.linalg.norm(x_old)

            if k%1 == 0:
                self.log2(x_old, grad, x, k, tol_g_val, np.linalg.norm(x - x_old), f.eval(x))

            k += 1

        return xs

    def log(self, curr_iter, error, loss):
        """
        """
        print("-----------------------------------")
        print("\n Iter: ", curr_iter)
        print("\n error: %s" % error)
        print("\n loss: %s" % loss)

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
