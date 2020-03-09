import numpy as np


class Rosenbrock:
    """
    Class for evaluating a function including the gradient and hessian matrix
    at a given point x

    '''

    Atributes
    ---------
    n : int
        size of sum in Rosenbrock
    """

    def __init__(self, n):
        """
        Sets size of sum in Rosenbrock
        """

        self.n = n

    def get_size(self):
        """Gets size of sum in Rosenbrock
        """

        return self.n

    def eval(self, x):
        """Evaluates function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which function is going to be evaluated
        """

        sum = 0

        for i in range(self.n-1):

            sum += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2

        return sum

    def gradient(self, x):
        """Evaluates gradient of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        grad = []

        for i in range(self.n-1):

            grad.append(-400*(x[i]*x[i+1] - x[i]**3) + 2*(x[i] - 1))

        grad.append(200*(x[self.n-1] - x[self.n-2]**2))

        return np.array(grad)

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
