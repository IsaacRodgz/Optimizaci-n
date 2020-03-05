import argparse
from test_function import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Step methods')

    parser.add_argument('-s', '--step', default="cubic", type=str, help='Gradient step size method to use: cubic, barzilai, zhang')

    parser.add_argument('-p', '--point', default="const", type=str, help='Type of starting point x: const, rand')

    parser.add_argument('-f', '--function', default="rosenbrock", type=str, help='Function to minimize: rosenbrock, wood, mnist')

    args = parser.parse_args()

    run(args.step, args.point, "gd", args.function)
