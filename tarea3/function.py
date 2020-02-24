import numpy as np


class Function:
    """
    Class for evaluating a function including the gradient and hessian matrix
    at a given point x

    '''

    Atributes
    ---------
    y : numpy.array
        used to evaluate the function at a given point x
    lambda_p : float
        parameter used to evaluate the function at a given point x
    n : int
        size of vector y
    """

    def __init__(self, lambda_p):
        """
        Reads file containing y vector used to evaluate the function.
        Sets attributes y, lambda_p and n.
        """

        with open('yk.txt', 'r') as f:
            y = f.read().replace('\n', '').split(',')[1:]

        self.y = np.array([float(num) for num in y])
        self.lambda_p = lambda_p
        self.n = len(y)

    def get_size(self):
        """Gets size of vector y
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
            sum += (x[i] - self.y[i])**2 + self.lambda_p*(x[i+1] - x[i])**2

        sum += (x[self.n-1] - self.y[self.n-1])**2

        return sum

    def gradient(self, x):
        """Evaluates gradient of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        grad = np.zeros(self.n)
        grad[0] = 2*(x[0] - self.y[0]) - 2*self.lambda_p*(x[1] - x[0])

        for i in range(1, self.n-1):
            grad[i] = 2*(x[i] - self.y[i]) \
                      - 2*self.lambda_p*(x[i+1] - x[i]) \
                      + 2*self.lambda_p*(x[i] - x[i-1])

        grad[self.n-1] = 2*(x[self.n-1] - self.y[self.n-1]) \
            + 2*self.lambda_p*(x[self.n-1] - x[self.n-2])

        return grad

    def hessian(self, x):
        """Evaluates hessian of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which hessian is going to be evaluated
        """

        hessian = np.zeros((self.n, self.n))

        hessian[0][0] = 2 + 2*self.lambda_p
        hessian[0][1] = -2*self.lambda_p
        hessian[1][0] = hessian[0][1]

        for i in range(1, self.n-1):
            hessian[i][i] = 2 + 4*self.lambda_p
            hessian[i][i+1] = -2*self.lambda_p
            hessian[i+1][i] = hessian[i][i+1]

        hessian[self.n-1][self.n-1] = 2 + 2*self.lambda_p

        return hessian
