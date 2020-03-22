from dogleg import Dogleg
from lstr import LSTR
from rbf import RBF
import numpy as np
import os


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


def train(optim_params, f_params, iters):

    mu_old = [0, 0]
    tau = f_params['tau']

    if optim_params["method"] == "dogleg":
        alg = Dogleg()
    elif optim_params["method"] == "lstr":
        alg = LSTR()

    f1 = RBF(f_params['basis_size'], f_params['sigma'], f_params['y'][0].flatten())
    f2 = RBF(f_params['basis_size'], f_params['sigma'], f_params['y'][1].flatten())

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
                                        "1")
        f_params['mu'][1] = alg.iterate(f_params['mu'][1],
                                        optim_params["mxitr"],
                                        optim_params["tol_g"],
                                        optim_params["tol_x"],
                                        optim_params["tol_f"],
                                        f2,
                                        "1")

        print("\nCurrent iter: {0}".format(i+1))

        norm_mu1 = np.linalg.norm(mu_old[0]-f_params['mu'][0])
        norm_mu2 = np.linalg.norm(mu_old[1]-f_params['mu'][1])

        if norm_mu1 < f_params['epsilon'] or norm_mu2 < f_params['epsilon']:

            print("Stop at iteration {0}".format(i+1))
            break;


if __name__ == '__main__':

    histograms = read_histograms("histograms")

    f_params = {'sigma': 10,
                'basis_size': 10,
                'epsilon': 0.1,
                'mu': [],
                'alpha': [0, 0],
                'y': histograms,
                'tau': 1.2}
    f_params['mu'].append(np.linspace(0, 255, f_params['basis_size']))
    f_params['mu'].append(np.linspace(0, 255, f_params['basis_size']))

    optim_params = {'method': "dogleg",
                    'mxitr': 10,
                    'tol_g': 1e-8,
                    'tol_x': 1e-8,
                    'tol_f': 1e-8}

    train(optim_params, f_params, 100)
