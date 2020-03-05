import argparse
from test_function import run


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Newton method')

    parser.add_argument('-s', '--step', default="fijo", type=str, help='Gradient step size method to use: fijo, hess, back')

    args = parser.parse_args()

    run(args.step)
