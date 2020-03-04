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

        self.n = self.train_data.shape[1]

    def get_size(self):
        """Gets size of flattened image
        """

        return self.n

    def sigmoid(self, x, beta):

        return 1/(1 + np.exp(-x.dot(beta)))

    def eval(self, beta):
        """Evaluates cost function

        Parameters
        ----------
        beta : numpy.array
            Weights of logistic regression
        """

        sum = 0

        for i in range(self.n):

            yi = self.train_labels[i]
            pi = self.sigmoid(self.train_data[i], beta)

            sum += yi*np.log10(pi) + (1-yi)*np.log10(1-pi)

        return sum

    def gradient(self, beta):
        """Evaluates gradient of cost function

        Parameters
        ----------
        beta : numpy.array
            Weights of logistic regression
        """

        grad = []

        for i in range(self.n):
            grad.append(self.train_labels[i] - self.sigmoid(self.train_data[i], beta))

        return np.array(grad)

    def error(self, beta):
        """Evaluates classification error on all the dataset

        Parameters
        ----------
        x : numpy.array
            Point at which gradient is going to be evaluated
        """

        sum = 0

        for i in range(self.n):

            yi = self.train_labels[i]
            pi = round(self.sigmoid(self.train_data[i], beta))

            sum += abs(pi - yi)

        return sum/self.n

"""
def p(x, beta, beta0):

    return 1/(1 + np.exp(-x.dot(beta) - beta0))

with gzip.open('mnist.pkl.gz','rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, val_set, test_set = u.load()

print(train_set[0].shape, train_set[1].shape)
print(val_set[0].shape, val_set[1].shape)
print(test_set[0].shape, test_set[1].shape)

zeros = np.where(train_set[1] == 0)[0]
ones = np.where(train_set[1] == 1)[0]
nums_index = np.concatenate((zeros, ones))
np.random.shuffle(nums_index)

train_images = train_set[0][nums_index]
train_labels = train_set[1][nums_index]

print(train_images.shape)
print(train_labels.shape)
idx = 1 # index of the image
im = train_images[idx].reshape(28, -1)
plt.imshow(im, cmap=plt.cm.gray)
plt.show()
print('Label: ', train_labels[idx])
"""
