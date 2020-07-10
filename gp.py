from create_data import create_ellipsis, geometric_matrix
import sys
import numpy as np
from gp_backend import GaussianProcessEuclidean, GaussianProcessVectorial, reconstruction_grid
from plotting import plot_emiss_signal, data_heatmap
import matplotlib.pyplot as plt

def compute_abs_error(measurement, inf_err_mask, best_data_post_std, post_data_mean):
    filtered_measurement = measurement[inf_err_mask]
    filtered_data = post_data_mean[inf_err_mask]
    assert len(filtered_data) == len(filtered_measurement)
    dif = filtered_data - filtered_measurement
    dif_squared = dif ** 2
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    sqrt_dif = np.sqrt(dif_squared)
    
    dif = np.abs(filtered_data - filtered_measurement)
    norm = dif/np.maximum(np.finfo(np.float32).eps, np.abs(filtered_measurement))
    return norm

def main(args):
    num_ellipses = int(sys.argv[1])
    S = 208
    max_density = 100.
    randomize_params = True
    img_dims = [32,32]
    noise_scale = 0.
    geo_mat, geo_mat_plot = geometric_matrix(img_dims, 1)
    # plt.imshow(geo_mat_plot.T, origin='lower')
    # plt.show()
    densities = []
    projections = []
    density_args = []
    for e in range(int(num_ellipses)):
        density, projection_geo_mat, Px, Py, ellipsis_args = create_ellipsis(S, max_density, img_dims, geo_mat, randomize_params, noise_scale)
        densities.append(density)
        projections.append(projection_geo_mat)
        density_args.append(ellipsis_args)
    densities = np.asarray(densities)
    projections = np.asarray(projections)
    density_args = np.asarray(density_args)
    print(densities.shape, projections.shape, density_args.shape)
    
    sigma_fs = np.asarray([10, 50, 100])
    sigma_l1s = np.asarray([.1, .15, .2])
    sigma_l2s = np.asarray([.1, .15, .2])
    sigma_errs = np.asarray([.001, ])
    
    density_shape = (densities.shape[1], densities.shape[2])
    
    pr_mean = np.zeros(density_shape).flatten()
    # print(densities.shape, pr_mean.shape)
    # exit(0)
    grid = reconstruction_grid(densities.shape[1], densities.shape[2])
    # print(grid.shape)
    # exit(0)
    gp = GaussianProcessEuclidean(geo_matrix = geo_mat,
                                  prior_mean = pr_mean,
                                  grid=grid,
                                  kernel='exponential',
                                  sigma_fs=sigma_fs,
                                  sigma_xs=sigma_l1s,
                                  sigma_errs=sigma_errs)
    # 
    # gp = GaussianProcessVectorial(geo_matrix = geo_mat,
    #                               prior_mean = pr_mean,
    #                               grid=grid,
    #                               kernel='exponential',
    #                               sigma_fs=sigma_fs,
    #                               sigma_l1s=sigma_l1s,
    #                               sigma_l2s=sigma_l2s,
    #                               sigma_errs=sigma_errs)
  
    
    
    
    
    
    
    reconstruction_means = []
    reconstruction_stds = []
    data_posterior_stds = []
    reconstruction_hps = []
    reconstruction_sigma_errs = []
    reconstruction_sigma_fs = []
    reconstruction_sigma_l1s = []
    reconstruction_sigma_l2s = []
    best_sigmas_grid = []
    times = []
    # shots = get_shots()
    
    
    posterior_data_means = []
    posterior_data_stds = []
    
    # exp_hps = {'sigma_fs':sigma_fs, 'sigma_xs':sigma_xs, 'sigma_errs':sigma_errs}
    # save_dic(exp_hps, exp + '/hp_params')
    mse_errs = []
    single_multiclasses = []
    comput_times = []
    total_samples = 0
    
    # for means_id, (measurement, errors) in enumerate(zip(measurement_data, error_data)):
    for means_id, measurement in enumerate(projections):
        inf_err_mask = np.ones(len(measurement)).astype(np.bool) #treat all measurements as noisy, but reliable
        # print(inf_err_mask.shape)
        # exit(0)
        best_post_mean, best_post_cov, best_sigma_d, mll, best_hps, best_sigma_grid, single_multiclass,t_delta  = gp.marginalize_and_calculate_hps(means_id,
                                                                                                                        measurement,
                                                                                                                        inf_err_mask)
        best_sigma_err_cat = best_hps[0]
        best_sigma_f_cat = best_hps[1]
        best_sigma_l1_cat = best_hps[2]
        # best_sigma_l2_cat = best_hps[3]
        comput_times.append(t_delta)
        best_data_post_std = np.sqrt(np.diagonal(best_sigma_d))
        # data_posterior_stds.append(best_data_post_std)
        # print(single_multiclass)
        print(means_id+1, '/', len(projections), '.', 'mll: %.3f' % (mll,), 'HP ids:', 
              'err:', np.round(sigma_errs[np.where(best_sigma_err_cat==1)],4),
              'f:', np.round(sigma_fs[np.where(best_sigma_f_cat==1)],4),
              'l1:', np.round(sigma_l1s[np.where(best_sigma_l1_cat==1)],4),
              # 'l2:', np.round(sigma_l2s[np.where(best_sigma_l2_cat==1)],4)
              # int(np.argwhere(single_multiclass==1)[0])
              # 'used measurements:', np.sum(inf_err_mask.astype(np.int))
              )
        reconstruction_means.append(best_post_mean)
        reconstruction_stds.append(np.sqrt(np.diag(best_post_cov)))
        reconstruction_hps.append(best_hps)
        reconstruction_sigma_errs.append(best_sigma_err_cat)
        reconstruction_sigma_fs.append(best_sigma_f_cat)
        reconstruction_sigma_l1s.append(best_sigma_l1_cat)
        # reconstruction_sigma_l2s.append(best_sigma_l2_cat)
        best_sigmas_grid.append(best_sigma_grid)
        # times.append(t)
        post_data_mean = geo_mat.dot(reconstruction_means[-1])
        posterior_data_means.append(post_data_mean)
        posterior_data_stds.append(best_data_post_std)
        single_multiclasses.append(single_multiclass)
        
        # old_rec = old_recs[:,:,means_id+500]
        # pdf_handler = PdfPages(exp + '/' + str(shot_id) + 'plots.pdf')
        original_profile = densities[means_id]
        plot_emiss_signal(reconstruction_means[-1], reconstruction_stds[-1],
                      post_data_mean, posterior_data_stds[-1],
                      measurement,
                      inf_err_mask,
                      # pdf_handler,
                      # sigma_errs[best_sigma_err_ind], sigma_fs[best_sigma_f_ind], sigma_xs[best_sigma_x_ind],
                        best_hps,mll, density_shape, original_profile
                      )
        sys.stdout.flush()
        
        mse_errs_this = compute_abs_error(measurement, inf_err_mask, 3*best_data_post_std, post_data_mean)
        mse_errs.extend(mse_errs_this)
    np.save('./posterior_data_means', np.asarray(posterior_data_means))
    
if __name__ == '__main__':
    main(sys.argv)
    