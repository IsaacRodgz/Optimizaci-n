import numpy as np
from scipy.sparse import csr_matrix


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
        self.dimx, self.dimy = self.g.shape

    def eval(self, x):
        """Evaluates function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which function is going to be evaluated
        """

        val = 0

        row = self.dimx
        col = self.dimy

        neighbours = [(1,0), (-1,0), (0,1), (0, -1)]
        x = x.reshape((self.dimx, self.dimy))

        for i in range(row):
            for j in range(col):
                val_neigh = 0
                for neigh in neighbours:
                    if not(i+neigh[0] < 0 or i+neigh[0] >= row) and not(j+neigh[1] < 0 or j+neigh[1] >= col):
                        indx = i+neigh[0]
                        indy = j+neigh[1]
                        val_neigh += (x[i][j] - x[indx][indy])**2
                val += self.lmbd*val_neigh

        val += np.sum((x-self.g)**2)

        return val

    def gradient(self, x):
        """Evaluates gradient of function at point x

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        row = self.dimx
        col = self.dimy

        neighbours = [(1,0), (-1,0), (0,1), (0, -1)]
        x = x.reshape((self.dimx, self.dimy))
        grad = np.zeros(row*col)

        for i in range(row):
            for j in range(col):
                grad[i*col+j] = 2*(x[i][j]-self.g[i][j])
                for neigh in neighbours:
                    if not(i+neigh[0] < 0 or i+neigh[0] >= row) and not(j+neigh[1] < 0 or j+neigh[1] >= col):
                        indx = i+neigh[0]
                        indy = j+neigh[1]
                        grad[i*col+j] += 4*self.lmbd*(x[i][j] - x[indx][indy])

        return grad

    def get_Q(self):
        """Returns matrix of quadratic form
        """

        row = self.dimx
        col = self.dimy

        H = np.zeros((row*col, row*col))

        for i in range(0, row*col):
            H[i][i] = 2 + 16*self.lmbd
            if (i-1)%row == 0:
                H[i-1][i] = H[i+1][i] = -4*self.lmbd
                H[i][i-1] = H[i][i+1] = -4*self.lmbd

        for i in range(0, row*col-3):
            H[i+3][i] = -4*self.lmbd

        for i in range(0, row*col-3):
            H[i][i+3] = -4*self.lmbd

        return csr_matrix(H)
