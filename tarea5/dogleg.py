import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError


class Dogleg:

    def iterate(self, x0, mxitr, tol_g, tol_x, tol_f, f, step):

        k = 0  # Start iteration at 0
        x = x0  # Start with initial point
        xs = []  # List to save points
        eta = 0.1
        delta0 = 100
        delta = 1

        # Iterate while max. num. of iters has not been reached
        while k < mxitr:

            xs.append(x)  # Save current point
            grad = f.gradient(x)

            # Calculate step size depending on value of step
            if step == '1':
                pk = self.get_step_cauchy(x, f, delta)
            elif step == '2':
                pk = self.get_step_norm(x, f, delta)
            elif step == '3':
                pk = self.get_step_pd(x, f, delta)
            else:
                print("\n Paso no valido")
                break
            # Evaluate quality of the quadratic model
            rho_k = (f.eval(x)-f.eval(x+pk))/(f.mk(x)-f.mk(x, pk))
            # Update radius of confidence region
            if rho_k < 0.25:
                delta *= 0.25
            else:
                if rho_k > 0.75 and np.linalg.norm(pk) == delta0:
                    delta = min(2*delta, delta0)

            # Make step forward gradient direction
            print("rho_k: ", rho_k)
            print("\n")
            if rho_k > eta:
                x_old = x
                x = x + pk

            # Calculate different tolerance criteria
            tol_x_val = np.linalg.norm(x - x_old) / max(1.0, np.linalg.norm(x_old))
            tol_f_val = np.absolute(f.eval(x) - f.eval(x_old)) / max(1.0, np.absolute(f.eval(x_old)))
            tol_g_val = np.linalg.norm(x_old)

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

    def get_step_cauchy(self, x, f, delta):
        grad = f.gradient(x)
        hess = f.hessian(x)

        # Cauchy step
        return self.get_cauchy_step(x, f, grad, hess, delta)

    def get_step_norm(self, x, f, delta):
        grad = f.gradient(x)
        hess = f.hessian(x)

        p_b = inv(hess).dot(grad)

        if np.linalg.norm(p_b) <= delta:
            return p_b
        else:
            return self.get_cauchy_step(x, f, grad, hess, delta)

    def get_step_pd(self, x, f, delta):
        grad = f.gradient(x)
        hess = f.hessian(x)

        # Check if hess matrix is positive definite
        is_pd = False
        try:
            cholesky(hess)
            is_pd = True
        except LinAlgError:
            pass

        if is_pd:  # Dogleg step
            return self.get_dogleg_step(x, f, grad, hess, delta)
        else:  # Cauchy step
            return self.get_cauchy_step(x, f, grad, hess, delta)

    def get_dogleg_step(self, x, f, grad, hess, delta):
        # Step in gradient direction without restriction
        alpha_u = (grad.dot(grad))/(grad.dot(hess).dot(grad))
        p_u = -alpha_u*grad

        # Complete step with hess positive definite
        p_b = -inv(hess).dot(grad)

        # Check if complete step is inside confidence region
        if np.linalg.norm(p_b) <= delta:
            return p_b
        # Calculate intercept between Dogleg trajectory and confidence region
        elif np.linalg.norm(p_u) >= delta:
            return (delta/np.linalg.norm(p_u))*p_u
        else:
            diff = p_b-p_u
            a = diff.dot(diff)
            b = 2*p_b.dot(diff)
            c = p_u.dot(p_u)-delta**2
            tau = 1 + (-b + np.sqrt(b**2-4*a*c))/(2*a)

            if tau <= 1:  # 0 <= tau <= 1
                return tau*p_u
            else:  # 1 < tau <= 2
                return p_u+(tau-1)*diff

    def get_cauchy_step(self, x, f, grad, hess, delta):
        # Point that minimizes lineal version
        alpha_s = delta/np.linalg.norm(grad)
        p_s = -alpha_s*grad

        prod = grad.dot(hess).dot(grad)

        if prod <= 0:
            tau = 1
        else:
            tau = min(1, (np.linalg.norm(grad)**3)/(delta*prod))

        return tau*p_s

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
