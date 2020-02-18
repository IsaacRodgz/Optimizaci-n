import argparse
from test_rosenbrock_n2 import run_ros2
from test_rosenbrock_n100 import run_ros100
from test_wood import run_wood

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gradient descent')

    parser.add_argument('-s', '--step', default="fijo", type=str, help='Gradient step size method to use: fijo, hess, back')

    parser.add_argument('-p', '--point', default="const", type=str, help='Type of starting point x: const, rand')

    parser.add_argument('-f', '--func', default="ros2", type=str, help='Test function to use: ros2, ros100, wood')

    args = parser.parse_args()

    function_dict = {"ros2" : run_ros2, "ros100" : run_ros100, "wood" : run_wood}

    function_dict[args.func](args.step, args.point)
