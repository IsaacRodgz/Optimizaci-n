import numpy as np
import matplotlib.pyplot as plt
from NCG import NCG
from Rosenbrock import Rosenbrock
from Wood import Wood


def run(point_type, function, beta_type):

    # Validate function
    if function == "rosenbrock":
        n = 100
        f = Rosenbrock(n)
    elif function == "wood":
        n = 4
        f = Wood()
    else:
        print("\n Error, invalid function")
        quit()

    # Initial point
    if point_type == "const":
        if function == "rosenbrock":
            x0 = np.ones(n)
            x0[0] = -1.2
            x0[-2] = -1.2
        else:
            x0 = np.array([-3, -1, -3, -1])
    elif point_type == "rand":
        x0 = np.random.uniform(-1, 1, n)
    else:
        print("\n Error, invalid type of point")
        quit()

    # Estimate minimum point through optimization method chosen
    alg = NCG()

    mx_iter = 10000

    xs, fs, gs = alg.iterate(x0, f, mx_iter, 1e-8, beta_type)

    # Print point x found
    #print("\nf(x) =  ", f.eval(xs[-1]))

    # Plot f(x) through iterations
    plt.plot(np.array(range(len(fs))), fs)
    plt.legend(['f(x)'], loc = 'best')
    plt.xlabel("iteration")
    plt.show()

    # Plot ||gradient(x)|| through iterations
    plt.plot(np.array(range(len(gs))), gs)
    plt.legend(['grad(x)'], loc = 'best')
    plt.xlabel("iteration")
    plt.show()
