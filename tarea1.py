import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def f1(x1, x2):

    return x1**2 - x2**2

def f2(x1, x2):

    return 2*x1*x2

def axes():

    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

def main():

    mpl.rcParams['lines.color'] = 'k'
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

    x = np.linspace(-9, 9, 400)
    y = np.linspace(-5, 5, 400)
    x, y = np.meshgrid(x, y)

    a = .3
    axes()
    plt.contour(x, y, f2(x, y), [16], colors='k')
    plt.contour(x, y, f1(x, y), [12], colors='k')
    plt.show()

if __name__ == '__main__':
    main()
