import numpy as np


class RBF:
    """
    Class for evaluating a function including the gradient and hessian matrix
    at a given point x

    '''

    Atributes
    ---------
    n : int
        Number of basis elements (gaussians)
    sigma: float
        Variance of each gaussian
    c : numpy.array
        Points to fit with gaussians
    alpha : numpy.array
        Vector of coefficients that weight the sum of gaussians
    Phi : numpy.matrix
        Kernel matrix of gaussian components
    """

    def __init__(self, n, sigma, c):
        """
        Sets size of sum in Rosenbrock
        """

        self.n = n
        self.sigma = sigma
        self.c = c
        self.alpha = None
        self.Phi = None

    def get_size(self):
        """Gets size of sum in Rosenbrock
        """

        return self.n

    def eval_h(self, mu):
        """Evaluates RBF function combining Phi according to weights alpha

        Parameters
        ----------
        mu : numpy.array
            vector of means of each gaussian
        """

        return self.get_kernel(mu)@self.alpha

    def eval(self, mu):
        """Evaluates function to minimize (h(c) - rbf(c))**2

        Parameters
        ----------
        mu : numpy.array
            vector of means of each gaussian
        """

        diff = self.c - self.get_kernel(mu)@self.alpha

        return np.sum(diff**2)

    def get_kernel(self, mu):
        """Build kernel matrix

        Parameters
        ----------
        c : numpy.array
            Points where the gaussians are going to be evaluated
        mu : numpy.array
            Mean of each gaussian
        """

        c_matrix = np.tile(self.c, (self.n,1)).T
        r = c_matrix-mu
        th = 1./self.sigma
        return np.exp(-0.5*(th*r)**2)

    def set_phi(self, phi):
        self.Phi = phi

    def set_alpha(self, alpha):
        self.alpha = alpha

    def gradient(self, x):
        """Evaluates gradient of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        grad = np.zeros(self.n)
        K = self.get_kernel(x)
        alpha = self.alpha.flatten()
        diff = K@alpha - self.c

        for i in range(self.n):
            phi_grad = alpha[i]*K[:,i]*(self.c-x[i])
            grad[i] = diff.dot(phi_grad)

        return (2/self.sigma**2)*grad

    def hessian(self, x):
        """Evaluates hessian of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which hessian is going to be evaluated
        """

        hessian = np.zeros((self.n, self.n))
        alpha = self.alpha.flatten()
        K = self.get_kernel(x)
        diff = K@alpha - self.c

        for i in range(self.n):
            for j in range(self.n):
                phi_grad_i = (alpha[i]/self.sigma**2)*K[:,i]*(self.c-x[i])
                phi_grad_j = (alpha[j]/self.sigma**2)*K[:,j]*(self.c-x[j])

                if i==j:
                    phi_grad_cross = (1/self.sigma**2)*K[:,i]*((1/self.sigma**2)*(self.c-x[i])**2-1)
                else:
                    phi_grad_cross = np.zeros(self.c.shape[0])

                hessian[i][j] = 2*(phi_grad_i.dot(phi_grad_j) + diff.dot(phi_grad_cross))

        return hessian

    def mk(self, x, *args):
        if len(args)>0:
            d = args[0]
            return self.eval(x)+self.gradient(x).dot(d)+0.5*d.dot(self.hessian(x)).dot(d)
        return self.eval(x)

    def mk_grad(self, x, d):
        return self.gradient(x) + self.hessian(x).dot(d)
