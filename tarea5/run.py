import argparse
from test_function import run


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Newton method')

    parser.add_argument('-s', '--step', default="fijo", type=str, help='Gradient step size method to use: fijo, hess, back')

    parser.add_argument('-p', '--point', default="const", type=str, help='Type of starting point x: const, rand')

    args = parser.parse_args()

    run(args.step, args.point)
