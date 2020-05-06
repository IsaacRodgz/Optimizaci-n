import numpy as np
import matplotlib.pyplot as plt
from DFP import DFP
from BFGS import BFGS
import time


def eval_rosenbrock(x):
    n = x.shape[0]
    sum = 0

    for i in range(n-1):

        sum += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2

    return sum


def gradient_rosenbrock(x):
    n = x.shape[0]
    grad = []

    for i in range(n-1):

        grad.append(-400*(x[i]*x[i+1] - x[i]**3) + 2*(x[i] - 1))

    grad.append(200*(x[n-1] - x[n-2]**2))

    return np.array(grad)


def eval_wood(x):
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


def gradient_wood(x):
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


def run(function, method):

    # Validate function
    if function == "rosenbrock":
        f = eval_rosenbrock
        g = gradient_rosenbrock
        n = 100
    elif function == "wood":
        f = eval_wood
        g = gradient_wood
        n = 4
    else:
        print("\n Error, invalid function")
        quit()

    # Estimate minimum point through optimization method chosen
    if method == "dfp":
        alg = DFP()
    elif method == "bfgs":
        alg = BFGS()
    else:
        print("\n Error, invalid optimization method")
        quit()

    mx_iter = 10000
    tol = 1e-6
    runs = 30

    exec_times = []
    gradient_run = []
    iters_run = []

    for i in range(runs):
        start_time = time.time()
        x0 = np.random.uniform(0, 1, n)
        xs, fs, gs = alg.iterate(x0, f, g, mx_iter, tol)
        exec_times.append(time.time() - start_time)
        iters_run.append(len(xs))
        gradient_run.append(gs[-1])

    print("\n Mean execution time: {}".format(np.mean(exec_times)))
    print(" Mean number of iterations: {}".format(np.mean(iters_run)))
    print(" Mean gradient norm: {}".format(np.mean(gradient_run)))

    '''
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
    '''
