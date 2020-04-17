import numpy as np


class Wood:
    """
    Class for evaluating a function including the gradient and hessian matrix
    at a given point x
    """

    def eval(self, x):
        """Evaluates function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which function is going to be evaluated
        """

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        return sum((
            100*(x1**2 - x2)**2,
            (x1-1)**2,
            (x3-1)**2,
            90*(x3**2 - x4)**2,
            10.1*((x2-1)**2 + (x4-1)**2),
            19.8*(x2-1)*(x4-1),
            ))

    def gradient(self, x):
        """Evaluates gradient of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        return np.array([
                400*x1*(x1**2-x2) + 2*(x1-1),
                -200*(x1**2-x2) + 20.2*(x2-1) + 19.8*(x4-1),
                2*(x3-1) + 360*x3*(x3**2-x4),
                -180*(x3**2-x4) + 20.2*(x4-1) + 19.8*(x2-1)
                ])

    def hessian(self, x):
        """Evaluates hessian of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which hessian is going to be evaluated
        """

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        hessian = np.zeros((4, 4))

        hessian[0][0] = 1200*x1**2 - 400*x2 + 2
        hessian[0][1] = -400*x1

        hessian[1][0] = -400*x1
        hessian[1][1] = 220.2
        hessian[1][3] = 19.8

        hessian[2][2] = 1080*x3**2 - 360*x4 + 2
        hessian[2][3] = -360*x3

        hessian[3][1] = 19.8
        hessian[3][2] = -360*x3
        hessian[3][3] = 200.2

        return hessian
