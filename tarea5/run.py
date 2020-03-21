import argparse
from test_function import run


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Newton method')

    parser.add_argument('-m', '--method', default="dogleg", type=str, help='Method to use: dogleg, lstr')

    parser.add_argument('-p', '--point', default="const", type=str, help='Type of starting point x: const, rand')

    args = parser.parse_args()

    run(args.method, args.point)
