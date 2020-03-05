import numpy as np
import matplotlib.pyplot as plt
from newton import Newton
from gradient_descent import GD
from Rosenbrock import Rosenbrock
from Wood import Wood
from mnist import MNIST


def run(step, point_type, method, function):

    # Validate function
    if function == "rosenbrock":
        n = 100
        f = Rosenbrock(n)
    elif function == "wood":
        n = 4
        f = Wood()
    elif function == "mnist":
        f = MNIST()
        n = f.get_size()
    else:
        print("\n Error, invalid function")
        quit()

    # Initial point
    if point_type == "const":
        if function == "rosenbrock":
            x0 = np.ones(n)
            x0[0] = -1.2
            x0[-2] = -1.2
        elif function == "mnist":
            x0 = np.ones(n)
        else:
            x0 = np.array([-3, -1, -3, -1])
    elif point_type == "rand":
        x0 = np.random.uniform(-1, 1, n)
    else:
        print("\n Error, invalid type of point")
        quit()

    mxitr = 50000  # Max number of iterations

    tol_g = 1e-8  # Tolerance for gradient

    tol_x = 1e-8  # Tolerance for x

    if function == "mnist":
        tol_f = 1e-3  # Tolerance for function
    else:
        tol_f = 1e-8  # Tolerance for function

    # Method for step update
    if step not in ["cubic", "barzilai", "zhang"]:
        print("\n Error, invalid step method")
        quit()

    # Estimate minimum point through optimization method chosen
    if method == "gd":
        alg = GD()
    elif method == "newton":
        alg = Newton()
    else:
        print("\n Error: Invalid optimization method: %s\n" % method)
        quit()

    xs = alg.iterate(x0, mxitr, tol_g, tol_x, tol_f, f, step, function)

    # Print point x found and function value f(x)
    # print("\nPoint x found: ", xs[-1])
    print("\nf(x) =  ", f.eval(xs[-1]))

    plt.plot(np.array(range(n)), xs[-1])
    plt.legend(['x*'], loc = 'best')
    plt.show()

    if function == "mnist":
        im = xs[-1][:-1].reshape(28, -1)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.show()
