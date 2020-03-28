import argparse
from test_function import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Step methods')

    parser.add_argument('-p', '--point', default="const", type=str, help='Type of starting point x: const, rand')

    parser.add_argument('-f', '--function', default="quadratic", type=str, help='Function to minimize: quadratic')

    parser.add_argument('-d', '--dim', default=128, type=int, help='Dimension of function matrix')

    parser.add_argument('-l', '--penalization', default=1, type=int, help='Regularization parameter')

    args = parser.parse_args()

    run(args.point, args.dim, args.penalization, args.function)
