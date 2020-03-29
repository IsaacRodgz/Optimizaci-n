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
                    if i+neigh[0] < 0 or i+neigh[0] >= row:
                        val_neigh += 0
                        continue
                    else:
                        indx = i+neigh[0]
                    if j+neigh[1] < 0 or j+neigh[1] >= col:
                        val_neigh += 0
                        continue
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

        row = self.dimx
        col = self.dimy

        neighbours = [(1,0), (-1,0), (0,1), (0, -1)]
        x = x.reshape((self.dimx, self.dimy))
        grad = np.zeros(row*col)

        for i in range(row):
            for j in range(col):
                val_neigh = 2*(x[i][j]-self.g[i][j])
                for neigh in neighbours:
                    if i+neigh[0] < 0 or i+neigh[0] >= row:
                        val_neigh += 0
                        continue
                    else:
                        indx = i+neigh[0]
                    if j+neigh[1] < 0 or j+neigh[1] >= col:
                        val_neigh += 0
                        continue
                    else:
                        indy = j+neigh[1]
                    val_neigh += 4*self.lmbd*(x[i][j] - x[indx][indy])
                grad[i*col+j] += (x[i][j] - self.g[i][j])**2 + self.lmbd*val_neigh

        return grad

    def get_Q(self):
        """Returns matrix of quadratic form
        """

        row = self.dimx
        col = self.dimy
        '''
        neighbours = [(1,0), (-1,0), (0,1), (0, -1)]
        x = x.reshape((self.dimx, self.dimy))
        H = np.zeros((row*col, row*col))

        index2rowcol = []
        k = 0
        for i in range(row):
            for j in range(col):
                index2rowcol.append((i,j))

        for i in range(row*col):
            for j in range(row*col):
                if i == j:
                    H[i][j] = 2+8*self.lmbd
                elif self.neigh(i, j, index2rowcol):

                    H[i][j] = -4*self.lmbd
                else:
                    H[i][j] = 0
        '''

        H = np.zeros((row*col, row*col))
        H[0][0] = 10
        H[0][1] = -4

        for i in range(1, row*col-1):
            H[i][i-1] = -4
            H[i][i] = 10
            H[i][i+1] = -4

        H[row*col-1][row*col-2] = -4
        H[row*col-1][row*col-1] = 10

        return H

    def neigh(self, i, j, index2rowcol):

        neighbours = [(1,0), (-1,0), (0,1), (0, -1)]
        current = index2rowcol[i]
        other = index2rowcol[j]

        for neigh in neighbours:
            if other[0] == current[0]+neigh[0] and other[1] == current[1]+neigh[1]:
                return True

        return False
