import matplotlib.pyplot as plt
import gzip, pickle
import numpy as np

class MNIST:
    """
    Class for evaluating a function including the gradient at a given point x
    The function is equivalent to binary cross-entropy
    '''

    Atributes
    ---------
    train_data : numpy.array
        Training data of mnist digit images
    train_labels : numpy.array
        Training labels of mnist digit images
    """

    def __init__(self):
        """
        Load MNIST dataset
        """

        with gzip.open('mnist.pkl.gz','rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            train_set, val_set, test_set = u.load()

        # Extract 0's and 1's from training dataset
        zeros = np.where(train_set[1] == 0)[0]
        ones = np.where(train_set[1] == 1)[0]
        nums_index = np.concatenate((zeros, ones))
        np.random.shuffle(nums_index)

        self.train_data  = train_set[0][nums_index]
        self.train_labels = train_set[1][nums_index]

        # Append 1 in last column
        p = np.zeros(self.train_data.shape[0])
        p = p.reshape((self.train_data.shape[0],1))
        self.train_data  = np.c_[self.train_data, p]

        # Extract 0's and 1's from test dataset
        zeros = np.where(test_set[1] == 0)[0]
        ones = np.where(test_set[1] == 1)[0]
        nums_index = np.concatenate((zeros, ones))
        self.test_data  = test_set[0][nums_index]
        self.test_labels = test_set[1][nums_index]

        # Append 1 in last column
        p = np.zeros(self.test_data.shape[0])
        p = p.reshape((self.test_data.shape[0],1))
        self.test_data  = np.c_[self.test_data, p]

        self.n = self.train_data.shape[1]
        self.data_size = self.train_data.shape[0]

    def get_size(self):
        """Gets size of flattened image
        """

        return self.n

    def sigmoid(self, x, beta):

        return 1/(1 + np.exp(-x.dot(beta)))

    def sigmoid(self, x, beta):

        return 1/(1 + np.exp(-x.dot(beta)))

    def eval(self, beta):
        """Evaluates cost function

        Parameters
        ----------
        beta : numpy.array
            Weights of logistic regression
        """

        yi = self.train_labels
        pi = self.sigmoid(self.train_data, beta)
        check1 = (pi == 0).astype(np.float)*(1e-15)
        check2 = ((1-pi) == 0).astype(np.float)*(1e-15)

        return yi.dot(np.log10(pi+check1)) + (1-yi).dot(np.log10((1-pi)+check2))

    def gradient(self, beta):
        """Evaluates gradient of cost function

        Parameters
        ----------
        beta : numpy.array
            Weights of logistic regression
        """

        yi = self.train_labels
        pi = self.sigmoid(self.train_data, beta)

        return self.train_data.T.dot(yi-pi)

    def error(self, beta):
        """Evaluates classification error on all the test dataset

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        sum = 0

        for i in range(self.test_data.shape[0]):

            yi = self.test_labels[i]
            pi = round(self.sigmoid(self.test_data[i], beta))

            sum += abs(pi - yi)

        return sum/self.test_data.shape[0]
