import argparse
from test_function import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Step methods')

    parser.add_argument('-f', '--function', default="rosenbrock", type=str, help='Function to minimize: rosenbrock, wood')

    parser.add_argument('-m', '--method', default="dfp", type=str, help='Quasi-Newton optimization method: dfp, bfgs')

    args = parser.parse_args()

    run(args.function, args.method)
