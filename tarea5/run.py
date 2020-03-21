import argparse
from test_function import run


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Newton method')

<<<<<<< HEAD
    parser.add_argument('-m', '--method', default="dogleg", type=str, help='Method to use: dogleg, lstr')
=======
    parser.add_argument('-s', '--step', default="3", type=str, help='Confidence region method: 1, 2, 3')
>>>>>>> 444a0f6d6ba14616b302882d5905fa7a2205c9ab

    parser.add_argument('-p', '--point', default="const", type=str, help='Type of starting point x: const, rand')

    args = parser.parse_args()

    run(args.method, args.point)
