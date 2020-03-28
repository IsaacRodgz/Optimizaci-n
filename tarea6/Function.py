import numpy as np


class Function:
    """
    Class for evaluating a function including the gradient of cuadratic function
    """

    def __init__(self, g, lmbd):
        """
        Sets quadratic form parameters
        """

        self.g = g
        self.lmbd = lmbd

    def eval(self, x):
        """Evaluates function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which function is going to be evaluated
        """

        val = 0

        row = self.g.shape[0]
        col = self.g.shape[1]

        neighbours = [(1,0), (-1,0), (0,1), (0, -1)]

        for i in range(row):
            for j in range(col):
                val_neigh = 0
                for neigh in neighbours:
                    if i+neigh[0] < 0 or i+neigh[0] >= row:
                        indx = i
                    else:
                        indx = i+neigh[0]
                    if j+neigh[1] < 0 or j+neigh[1] >= col:
                        indy = j
                    else:
                        indy = j+neigh[1]
                    val_neigh += (x[i][j] - x[indx][indy])**2
                val += (x[i][j] - self.g[i][j])**2 + self.lmbd*val_neigh

        return val

    def gradient(self, x):
        """Evaluates gradient of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        return self.g

    def get_Q(self):
        """Returns matrix of quadratic form
        """

        return self.g
