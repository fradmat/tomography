from create_data import create_ellipsis
import sys
import numpy as np
from gp_backend import GaussianProcessEuclidean

def main(args):
    num_ellipses = int(args[1])
    S = 207
    max_density = 100.
    randomize_params = True
    img_dims = [63,63]
    with_noise = True
    densities = []
    projections = []
    density_args = []
    for e in range(int(num_ellipses)):
        density, Px, Py, ellipsis_args = create_ellipsis(S, max_density, img_dims, randomize_params, with_noise)
        densities.append(density)
        projections.append(Py)
        density_args.append(ellipsis_args)
    densities = np.asarray(densities)
    projections = np.asarray(projections)
    density_args = np.asarray(density_args)
    print(densities.shape, projections.shape, density_args.shape)
    
    sigma_fs = np.asarray([500, 1000, 2500])
    sigma_xs = np.asarray([.15, .175, .2])
    sigma_errs = np.asarray([.5, .75, 1.])
    gp = GaussianProcessEuclidean(geo_matrix = geo_m, prior_mean = pr_mean, grid=grid, kernel='exponential', sigma_fs=sigma_fs,sigma_xs=sigma_xs, sigma_errs=sigma_errs)
    
if __name__ == '__main__':
    main(sys.argv)
    