from dogleg import Dogleg
from lstr import LSTR
from rbf import RBF
import numpy as np
import argparse
import cv2
import os


def plot_segments(f1, f2, f_params, img_path, num_bins, output_filename):
    image = cv2.imread(img_path)

    row = image.shape[0]
    col = image.shape[1]

    alpha0 = f_params['alpha'][0].reshape(f_params['basis_size'])
    alpha1 = f_params['alpha'][1].reshape(f_params['basis_size'])

    mu0 = f_params['mu'][0]
    mu1 = f_params['mu'][1]

    hist0 = f_params['y'][0].flatten()
    hist1 = f_params['y'][1].flatten()

    sigma = f_params['sigma']
    epsilon = 0.01

    image_segmented = np.zeros((row,col,3))

    for i in range(row):
        for j in range(col):

            bin_x = int(float(image[i,j,0])/256.0*num_bins)
            bin_y = int(float(image[i,j,1])/256.0*num_bins)
            bin_z = int(float(image[i,j,2])/256.0*num_bins)

            index = (num_bins**2)*bin_x + num_bins*bin_y + bin_z

            c0 = hist0[index]
            c1 = hist1[index]

            phi0 = np.exp(-0.5*(1/sigma**2)*(c0-mu0)**2)
            phi1 = np.exp(-0.5*(1/sigma**2)*(c1-mu1)**2)

            f_val0 = alpha0.dot(phi0)
            f_val1 = alpha0.dot(phi1)

            F_val0 = (f_val0 + epsilon)/(f_val0 + f_val1 + 2*epsilon)
            F_val1 = (f_val1 + epsilon)/(f_val0 + f_val1 + 2*epsilon)

            if F_val0 > F_val1:
                r, g, b = 255, 0, 0
            else:
                r, g, b = 0, 0, 255

            image_segmented[i, j, 0] = r
            image_segmented[i, j, 1] = g
            image_segmented[i, j, 2] = b

    cv2.imwrite(output_filename, image_segmented)


def plot_segments_true(histograms, img_path, num_bins, output_filename):
    image = cv2.imread(img_path)

    row = image.shape[0]
    col = image.shape[1]

    hist0 = histograms[0].flatten()
    hist1 = histograms[1].flatten()

    epsilon = 0.01

    image_segmented = np.zeros((row,col,3))

    for i in range(row):
        for j in range(col):

            bin_x = int(float(image[i,j,0])/256.0*num_bins)
            bin_y = int(float(image[i,j,1])/256.0*num_bins)
            bin_z = int(float(image[i,j,2])/256.0*num_bins)

            index = (num_bins**2)*bin_x + num_bins*bin_y + bin_z

            c0 = hist0[index]
            c1 = hist1[index]

            F_val0 = (c0 + epsilon)/(c0 + c1 + 2*epsilon)
            F_val1 = (c1 + epsilon)/(c0 + c1 + 2*epsilon)

            if F_val0 < F_val1:
                r, g, b = 255, 0, 0
            else:
                r, g, b = 0, 0, 255

            image_segmented[i, j, 0] = r
            image_segmented[i, j, 1] = g
            image_segmented[i, j, 2] = b

    cv2.imwrite(output_filename, image_segmented)


def read_histograms(path):
    files = list(os.walk(path))[0][2]
    files = ["/".join([path, file]) for file in files]

    histograms = []

    for file in files:
        with open(file, 'r') as file:
            hist = file.read().replace('\n', ' ').split()
            num_bins = int(hist[0])
            hist = hist[3:]
        hist = np.array([int(num) for num in hist])
        hist = hist.reshape((num_bins, num_bins, num_bins))

        histograms.append(hist)

    return histograms


def SolveRidgeRegression(X, y, tau):

    xtranspose = np.transpose(X)
    xtransx = np.dot(xtranspose, X)
    lamidentity = np.identity(xtransx.shape[0]) * tau
    matinv = np.linalg.inv(xtransx - lamidentity)
    xtransy = np.dot(xtranspose, y.flatten())
    beta = np.dot(matinv, xtransy)

    return beta


def build_design_matrix(mu, f1, f2):

    phi = []

    k1 = f1.get_kernel(mu[0])
    phi.append(k1)

    k2 = f2.get_kernel(mu[1])
    phi.append(k2)

    return phi


def train(optim_params, f_params, iters, f1, f2):

    mu_old = [0, 0]
    tau = f_params['tau']

    if optim_params["method"] == "dogleg":
        alg = Dogleg()
    elif optim_params["method"] == "lstr":
        alg = LSTR()

    for i in range(iters):

        mu_old[0] = f_params['mu'][0][:]
        mu_old[1] = f_params['mu'][1][:]

        f_params['Phi'] = build_design_matrix(f_params['mu'], f1, f2)

        #f_params['alpha'][0] = np.linalg.lstsq(f_params['Phi'][0], f_params['y'][0].flatten(), rcond=None)[0].reshape((f_params['basis_size'], 1))
        #f_params['alpha'][1] = np.linalg.lstsq(f_params['Phi'][1], f_params['y'][1].flatten(), rcond=None)[0].reshape((f_params['basis_size'], 1))
        f_params['alpha'][0] = SolveRidgeRegression(f_params['Phi'][0], f_params['y'][0], tau).reshape((f_params['basis_size'], 1))
        f_params['alpha'][1] = SolveRidgeRegression(f_params['Phi'][1], f_params['y'][1], tau).reshape((f_params['basis_size'], 1))

        f1.set_phi(f_params['Phi'][0])
        f1.set_alpha(f_params['alpha'][0])
        f2.set_phi(f_params['Phi'][1])
        f2.set_alpha(f_params['alpha'][1])

        f_params['mu'][0] = alg.iterate(f_params['mu'][0],
                                        optim_params["mxitr"],
                                        optim_params["tol_g"],
                                        optim_params["tol_x"],
                                        optim_params["tol_f"],
                                        f1,
                                        "3")[0][-1]
        f_params['mu'][1] = alg.iterate(f_params['mu'][1],
                                        optim_params["mxitr"],
                                        optim_params["tol_g"],
                                        optim_params["tol_x"],
                                        optim_params["tol_f"],
                                        f2,
                                        "3")[0][-1]

        print("\nCurrent iter: {0}".format(i+1))

        norm_mu1 = np.linalg.norm(mu_old[0]-f_params['mu'][0])
        norm_mu2 = np.linalg.norm(mu_old[1]-f_params['mu'][1])

        if norm_mu1 < f_params['epsilon'] or norm_mu2 < f_params['epsilon']:

            print("Stop at iteration {0}".format(i+1))
            break;


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image segmentation')
    parser.add_argument('-s', '--size', default=10, type=int, help='Number of radial basis functions')
    parser.add_argument('-m', '--method', default="dogleg", type=str, help='Optimization method: dogleg, lstr')
    parser.add_argument('-v', '--variance', default=10, type=float, help='Value of parameter sigma of rbf functions')
    parser.add_argument('-f', '--hist', default="histograms", type=str, help='Folder name containing histograms')
    parser.add_argument('-i', '--image', default="grave.bmp", type=str, help='Name of image to segment')
    parser.add_argument('-o', '--output', default="grave_segmented.png", type=str, help='Name of file of segmented image')
    parser.add_argument('-t', '--true', default="no", type=str, help='Option to segment image with true histogram or not: yes, no')
    args = parser.parse_args()

    histograms = read_histograms(args.hist)
    num_bins = histograms[0].shape[0]

    if args.true == "yes":
        plot_segments_true(histograms, args.image, num_bins, args.output)
    else:
        f_params = {'sigma': args.variance,
                    'basis_size': args.size,
                    'epsilon': 0.1,
                    'mu': [],
                    'alpha': [0, 0],
                    'y': histograms,
                    'tau': 1.2}
        f_params['mu'].append(np.linspace(0, 255, f_params['basis_size']))
        f_params['mu'].append(np.linspace(0, 255, f_params['basis_size']))

        optim_params = {'method': args.method,
                        'mxitr': 10,
                        'tol_g': 1e-8,
                        'tol_x': 1e-8,
                        'tol_f': 1e-8}

        f1 = RBF(f_params['basis_size'], f_params['sigma'], f_params['y'][0].flatten())
        f2 = RBF(f_params['basis_size'], f_params['sigma'], f_params['y'][1].flatten())

        train(optim_params, f_params, 100, f1, f2)

        plot_segments(f1, f2, f_params, args.image, num_bins, args.output)
