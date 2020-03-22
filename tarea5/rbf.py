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
        th = 1./(2.*self.sigma)
        return np.exp(-(th*r)**2)

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

        y = self.c
        y = y.reshape((y.shape[0], 1))
        Phi = self.Phi
        alpha = self.alpha
        sigma = self.sigma
        
        e = Phi@alpha-y
        prod1 = (y*np.ones((1,alpha.shape[0])) - (np.ones((y.shape[0],1))*x.T))
        prod2 = ((e*alpha.T)*Phi)*prod1

        return (2/sigma**2)*np.sum(prod2, axis = 0)

    def hessian(self, x):
        """Evaluates hessian of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which hessian is going to be evaluated
        """

        hessian = np.zeros((self.n, self.n))

        for i in range(self.n-1):

            hessian[i][i] = -400*x[i+1] + 1200*x[i]**2 + 2
            hessian[i][i+1] = -400*x[i]

        hessian[self.n-1][self.n-2] = -400*x[self.n-2]
        hessian[self.n-1][self.n-1] = 200

        return hessian

    def mk(self, x, *args):
        if len(args)>0:
            d = args[0]
            return self.eval(x)+self.gradient(x).dot(d)+0.5*d.dot(self.hessian(x)).dot(d)
        return self.eval(x)

    def mk_grad(self, x, d):
        return self.gradient(x) + self.hessian(x).dot(d)
