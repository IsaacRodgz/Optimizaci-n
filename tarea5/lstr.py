import numpy as np
from numpy.linalg import inv
from scipy.linalg import ldl


class LSTR:

    def iterate(self, x0, mxitr, tol_g, tol_x, tol_f, f):

        k = 0  # Start iteration at 0
        x = x0  # Start with initial point
        x_old = np.zeros(x.shape[0])
        xs = []  # List to save points
        eta = 0.1
        delta0 = 10
        delta = 1

        # Iterate while max. num. of iters has not been reached
        while k < mxitr:

            xs.append(x)  # Save current point
            grad = f.gradient(x)
            # Calculate step size depending on value of msg
            pk = self.get_step(x, f, delta, 20)
            # Evaluate quality of the quadratic model
            rho_k = (f.eval(x)-f.eval(x+pk))/(f.mk(x)-f.mk(x, pk))
            # Update radius of confidence region
            if rho_k < 0.25:
                delta *= 0.25
            else:
                if rho_k > 0.75 and np.linalg.norm(pk) == delta0:
                    delta = np.min(2*delta, delta0)

            # Make step forward gradient direction
            print("rho_k: ", rho_k)
            print("\n")
            if rho_k > eta:
                x_old = x
                x = x + pk

            # Calculate different tolerance criteria
            tol_x_val = np.linalg.norm(x - x_old) / max(1.0, np.linalg.norm(x_old))
            tol_f_val = np.absolute(f.eval(x) - f.eval(x_old)) / max(1.0, np.absolute(f.eval(x_old)))
            tol_g_val = np.linalg.norm(grad)

            if k%1 == 0:
                self.log_latex(x_old, grad, x, k, tol_g_val, np.linalg.norm(x - x_old), f.eval(x))

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

        return xs

    def get_step(self, x, f, delta, iters):

        i = 0
        z = np.zeros(x.shape[0])
        d = -f.mk_grad(x, z)

        while i < iters and np.linalg.norm(d) != 0:

            hess = f.hessian(x)

            if d.dot(hess).dot(d) < 0:
                a = np.linalg.norm(d)**2
                b = 2*z.dot(d)
                c = np.linalg.norm(z)**2 - delta**2
                tau = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

                return z + tau*d

            else:
                alpha = -(f.gradient(x).dot(d)+z.dot(hess).dot(z))/(d.dot(hess).dot(d))
                z_old = z
                z = z + alpha*d

                if np.linalg.norm(z) >= delta:
                    a = np.linalg.norm(d)**2
                    b = 2*z.dot(d)
                    c = np.linalg.norm(z_old)**2 - delta**2
                    tau = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

                    return z_old + tau*d

                d = -f.mk_grad(x, z)
                i += 1

        return z

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

    def log_latex(self, x_old, grad, x, curr_iter, tol_g_val, tol_x_val, tol_f_val):
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

        print("{0} & {1:.10E} & {2:.10E} & {3:.10E} \\\\".format(curr_iter, tol_x_val, tol_g_val, tol_f_val))
