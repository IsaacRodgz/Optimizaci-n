import argparse
from test_function import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Step methods')

    parser.add_argument('-p', '--point', default="const", type=str, help='Type of starting point x: const, rand')

    parser.add_argument('-f', '--function', default="rosenbrock", type=str, help='Function to minimize: rosenbrock, wood')

    parser.add_argument('-b', '--beta', default="fr", type=str, help='Function to minimize: fr, pr, hs')

    args = parser.parse_args()

    run(args.point, args.function, args.beta)
