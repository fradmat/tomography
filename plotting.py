import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib as mpl
# import matplotlib.cm as cmap
import seaborn as sns
from scipy import stats
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def plot_emiss_signal(emiss_mean, emiss_std, post_data_mean, post_data_std, measurement, inf_err_mask,
                      hps,mll, density_shape,
                      original_profile):
    X = np.linspace(0, 1, density_shape[0], endpoint=True)
    Y = np.linspace(0, 1, density_shape[0], endpoint=True)
    # print(emiss_mean.shape, density_shape, X.shape)
    # exit(0)
    
    fig = plt.figure(figsize=(6,5))
    # if not (old_rec is None):
    gs = gridspec.GridSpec(5, 6)
    # else:
    #     gs = gridspec.GridSpec(5, 4)
    current_cmap = plt.get_cmap('viridis')
    # current_cmap.set_bad(color='gray')
    
    norm_errors = ((emiss_std*3)/np.abs(emiss_mean)).reshape(density_shape)
    emiss_mean = emiss_mean.reshape(density_shape)
    # plt.rc('axes', axisbelow=True)
    
    ax_em_mean = fig.add_subplot(gs[0:3, 4:6])
    # mask = np.full(2400, np.nan)
    # mask[bdmat] = emiss_mean#[norm_errors<1]
    # em_mean_to_show = mask.reshape(60,40)
    # pl = ax_em_mean.imshow(em_mean_to_show, origin = 'lower', cmap=current_cmap, animated=True)
    
    pl = ax_em_mean.contourf(X, Y, emiss_mean.astype(np.float64), 20, cmap = current_cmap)
    
    ax_em_mean.grid(linewidth=.25)
    divider = make_axes_locatable(ax_em_mean)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(pl, cax=cax)
    
    
    
    ax_em_mean.set_title('Predicted distribution ($W/m^3$)')
    ax_em_mean.set_xticks([])
    ax_em_mean.set_yticks([])
    ax_em_mean.grid()
    # exit(0)
    
    ax_em_std = fig.add_subplot(gs[0:3, 2:4])
    
    # mask = np.full(2400, np.nan)
    # mask[bdmat] = np.clip(norm_errors, a_min=0, a_max=1)
    # mask[bdmat] = norm_errors
    # em_err_to_show = mask.reshape(60,40)
    norm_errors = np.clip(norm_errors, a_min=0, a_max=1)
    ax_em_std.grid(linewidth=.25)
    # steps = np.arange(12, step=2)/10
    # steps = np.arange(11)/10
    pl = ax_em_std.contourf(X, Y, norm_errors.astype(np.float64), cmap = current_cmap)
    print(np.min(norm_errors), np.max(norm_errors))
    # exit(0)
    ax_em_std.set_xticks([])
    ax_em_std.set_yticks([])
    # ax_em_std.set_title('Uncertainty ($W/m^3$)')
    # ax_em_std.set_xticks(X[[0, 10, 20, 30, 39]].round(2))
    # ax_em_std.set_xticks(X, minor=True)
    # # # for tick in ax_em_mean.xaxis.get_major_ticks():
    # # #     tick.label.set_fontsize(20) 
    # ax_em_std.set_yticks(Y[[0, 10, 20, 30, 40, 50, 59]].round(2))
    # ax_em_std.set_yticks(Y, minor=True)
    # # # for tick in ax_em_mean.yaxis.get_major_ticks():
    # # #     tick.label.set_fontsize(20) 
    # ax_em_std.set_xlabel('R(m)')
    # ax_em_std.set_ylabel('z(m)', rotation=90)
    
    # ax_em_std.set_facecolor('gray')
    # 
    ax_em_std.set_title('Reconstruction error (%)')
    # divider = make_axes_locatable(ax_em_std)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # plt.colorbar(pl, cax=cax, boundaries=np.append(steps, 1.2), ticks=steps,  extend='both',
    # #                             extendfrac='auto')
    # plt.colorbar(pl, cax=cax, boundaries=steps[:-1], ticks=steps,  extend='both',
    #                             extendfrac='auto')
    
    
    if not (original_profile is None):
        # print(old_rec.shape)
        ax_em_old = fig.add_subplot(gs[0:3, :2])
        # pl = ax_em_old.imshow(old_rec.reshape(120,80), origin = 'lower', cmap=plt.get_cmap('plasma'), animated=True)
        # X = np.linspace(-.55, .45 +.05, 80, endpoint=True)+1.7
        # Y = np.linspace(-.85, .65+ .05, 120, endpoint=True)-.15
        pl = ax_em_old.contourf(X, Y, original_profile.astype(np.float64), 20, cmap = current_cmap)
        ax_em_old.grid()
        divider = make_axes_locatable(ax_em_old)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pl, cax=cax)
        ax_em_old.set_title('Original distribution ($W/m^3$)')
        ax_em_old.set_xticks([])
        ax_em_old.set_yticks([])
    # 
    # 
    # # post_data_std_all = np.zeros(len(measurement))
    # # # post_data_std[np.logical_not(inf_err_mask)] = 0 # dummy values
    # # # print(post_data_std_all.shape, inf_err_mask.shape, post_data_std.shape)
    # # post_data_std_all[inf_err_mask] = post_data_std
    # # 
    # # norm_errors = post_data_std_all*3
    # # 
    # # # print(measurement.shape, measurement[inf_err_mask].shape)
    # # 
    # # ax_m_max = np.max(post_data_mean+norm_errors)
    # # ax_m_min = np.min(post_data_mean-norm_errors)
    # # print(np.max(measurement), np.max(measurement[inf_err_mask]), np.min(measurement), np.min(measurement[inf_err_mask]))
    # 
    ax_measurements = fig.add_subplot(gs[3:,:])
        
    # ax_measurements.set_ylim(bottom = ax_m_min, top = ax_m_max)
    filtered_measurement = measurement[inf_err_mask]
    # ax_measurements.plot(np.arange(208)[inf_err_mask],filtered_measurement, 'o', label='Data (Projection)')
    ax_measurements.plot(measurement, 'o', label='Data (Projection)')
    
    # if not (old_rec is None):
    #     ax_measurements.plot(old_rec_backproj, '--', label='MaxEntBackproj')
    # print(post_data_mean.shape, post_data_mean)
    # print(norm_errors.shape, norm_errors)
    ax_measurements.plot(post_data_mean, 'o', label='Backprojection', alpha=.33)
    ax_measurements.legend(loc='lower left')
    # ax_measurements.fill_between(np.arange(len(measurement)), measurement-norm_errors, measurement+norm_errors, where=inf_err_mask, alpha=.3)
    ax_measurements.set_title('SXR Measurements and Backprojection ($W/m^2$)')
    # ax_measurements.grid()
    # plt.suptitle('shot '+ str(shot_id) + ', t = ' +  str(time) + 
    #                  '. Used ' + str(np.sum(inf_err_mask.astype(np.int))) + ' measurements of 208' +
    #                  
    #                  ' '.join([hp_id + ': ' + str(round(hp_val,4)) for hp_id, hp_val in hps.items()] )
    #                  
    #                  # u'. \u03C3f: ' +
    #                  # str(round(sigma_f,4)) + u'. \u03C3x: ' +
    #                  # str(round(sigma_x,4)) + u'. \u03C3err: ' +
    #                  # str(round(sigma_err,4))
    #                  + u'. mll: ' +
    #                  
    #                  str(round(mll,4)), fontsize=10
    #                  )
    # 
    # plt.tight_layout(rect=[0, 0, 1, .95])
    # plt.grid()
    # pdf_handler.savefig()
    plt.show()
    
    plt.close()



def data_heatmap(best_sigmas_grid, sigma_fs, sigma_xs, sigma_errs, pdf_handler):
    best_sigmas_grid = np.asarray(best_sigmas_grid)
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2*len(sigma_errs))
    p = sns.color_palette('cool_d', 16)
    p.reverse()
    plt.rcParams["axes.grid"] = False
    for h, sigma_err in enumerate(sigma_errs):
        s = np.sum(best_sigmas_grid[:,h,:,:], axis=0).astype(int)
        print('for this sigma err, total is ', np.sum(s))
        heatmap = fig.add_subplot(gs[:, h*2:h*2+1])
        # pl = heatmap.imshow(s, origin = 'lower', cmap=plt.get_cmap('binary'), animated=True)
        # flatui = ["#7bc8f6", "#c071fe"]
        pl = heatmap.imshow(s, origin = 'lower', cmap= mpl.colors.ListedColormap(p), animated=True)
        # heatmap.grid()
        # We want to show all ticks...
        heatmap.set_xticks(np.arange(len(sigma_xs)))
        heatmap.set_yticks(np.arange(len(sigma_fs)))
        # ... and label them with the respective list entries
        heatmap.set_xticklabels(np.round(sigma_xs,4), fontsize=20)
        heatmap.set_yticklabels(np.round(sigma_fs,4), fontsize=20)
        plt.setp(heatmap.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        # print(best_sigma_grid_cat)
        # exit(0)
        # for edge, spine in heatmap.spines.items():
        #     spine.set_visible(False)
        # mpl.rc('font', **font)
        for i in range(len(sigma_fs)):
            for j in range(len(sigma_xs)):
                text = heatmap.text(j, i, s[i, j], ha="center", va="center", color="white", fontsize=5)

        heatmap.set_xticks(np.arange(s.shape[1]+1)-.5, minor=True)
        heatmap.set_yticks(np.arange(s.shape[0]+1)-.5, minor=True)

        heatmap.set_xlabel(r'$\sigma_x$', fontsize=30)
        heatmap.set_ylabel(r'$\sigma_F$', fontsize=30)
        heatmap.grid(which="minor", color="w", linestyle='-', linewidth=5)
        heatmap.tick_params(which="minor", bottom=False, left=False)
        heatmap.set_title(sigma_err)
        
    pdf_handler.savefig()
    plt.close()

def data_histogram(best_sigmas_grid, sigma_fs, sigma_xs, sigma_errs, fpath1):
    best_sigmas_grid = np.asarray(best_sigmas_grid)
    fig = plt.figure()
    # gs = gridspec.GridSpec(1,2*len(sigma_errs))
    # multi_class = np.zeros(len(sigma_fs) * len(sigma_xs) * len(sigma_errs))
    multi_class = np.reshape(np.copy(best_sigmas_grid), (best_sigmas_grid.shape[0], len(sigma_fs) * len(sigma_xs) * len(sigma_errs)))

    multi_class_cat = np.argmax(multi_class, axis=1)
    # print(multi_class_cat.shape, multi_class_cat[0])
    # exit(0)
    multi_class_sum = pd.Series({'category':multi_class_cat})
    # print(multi_class_sum.shape)
    # exit(0)
    g = sns.countplot(x='category', data=multi_class_sum, color =sns.color_palette("Blues").as_hex()[-2])
    rang = np.arange(0, 28, step=2)
    labels = [str(k) for k in rang ] + ['28',]
    plt.xticks(rang, labels=labels, rotation=45, size=15)
    for tick in g.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    # g.xaxis.set_minor_locator(AutoMinorLocator)
    plt.xlabel('Class', size=24)
    plt.ylabel('Count', size=24)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    figure = g.get_figure()
    figure.savefig(fpath1)
    

def plot_measurement(measurement, err, shot, time):
    filtered_measurement = measurement[err==1]
    # mpl.use('qt4agg')
    fs = mpl.rcParams['font.size']
    font = {'size': 30}
    fig = plt.figure()
    mpl.rc('font', **font)
    # plt.plot(measurement, 'o', )
    # print(filtered_measurement)
    # print(err)
    y_err = err
    # y_err = err/np.max(err[err!=float('inf')])
    # y_err[y_err == float('inf')] = 1
    # y_err[y_err == float('inf')] = 0
    y_err = np.logical_not(y_err.astype(bool)).astype(float)
    y_err *= 1e8
    measurement_norm = filtered_measurement/np.max(filtered_measurement)
    plt.plot(np.arange(208)[err==1], filtered_measurement, 'o')
    # markers, caps, bars = plt.errorbar(np.arange(208), measurement, yerr=y_err, fmt='o', ecolor='r')
    plt.ylim(-100, 1.1*np.max(filtered_measurement))
    # [bar.set_alpha(0.3) for bar in bars]
    # plt.title('Shot ' + str(shot) + ', t=' + str(time) + 's')
    print('Shot ' + str(shot) + ', t=' + str(time) + 's')
    plt.xlabel('Detector')
    plt.ylabel('Brightness (W/$m^2$)')
    
    # plt.grid()
    # plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
    font = {'size': fs}
    mpl.rc('font', **font)
    
    
def data_err_histo(mse_errs, fpath1, fpath2, err_type=''):
    print('plotting err histogram based on ', len(mse_errs), 'measurements, ', err_type)
    plt.rcParams["axes.grid"] = True
    n_bins = 50
    n_bins = np.arange(1.02, step=.02)
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 20}
    
    plt.rc('font', **font)
    fig = plt.figure()
    g = sns.distplot(mse_errs[mse_errs<=1], bins=n_bins)
    # plt.legend(['mean' + str(round(np.mean(mse_errs), 3)) + str(round(np.mean(np.log(mse_errs)), 3))])
    print(np.mean(mse_errs), np.max(mse_errs), np.min(mse_errs))
    # for k in []
    # g.axvline(x=0, color='r', linestyle='--')
    # g.set_xlim([0,1])
    g.set_xlabel('error')
    g.set_ylabel('PDF(error)')
    figure = g.get_figure()
    # plt.show()
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    figure.savefig(fpath1)
    
    print('----------------------------------------------')
    print('saved histo 1')
    percentiles = {}
    for thresh in [.1, .5, 1., 1e10]:
        percentiles[thresh] = np.sum(mse_errs<thresh)/len(mse_errs)
        print('----------------------------------------------')
        print('cum sum threshold:', str(thresh), percentiles[thresh])
    hist, bin_edges = np.histogram(mse_errs[mse_errs<=1], bins=n_bins)
    edge_len = bin_edges[1] - bin_edges[0]
    y = np.cumsum(hist)
    # print(y)
    y = y / y[-1]
    # print(y)
    # exit(0)
    fig = plt.figure()
    plt.plot(bin_edges[:-1]+ edge_len, y)
    kwargs = {'cumulative': True}
    n, bins, patches = plt.hist(mse_errs[mse_errs<=1], n_bins, density=True, 
                               cumulative=True, label='Empirical', color=sns.color_palette("Blues").as_hex()[-2], edgecolor='w', alpha=.4)
    # plt.axvline(x=0, color='r', linestyle='--')
    # plt.xlim([0, 1])
    # plt.show()
    plt.xlabel('error')
    plt.ylabel('CDF(error)')
    # plt.show()
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(fpath2)
    
    # percentile_99 = .99 * len(mse_errs)
    print('----------------------------------------------')
    print('saved histo 2')
    
    print('----------------------------------------------')
    print('mean backprojection err:', 100*np.mean(mse_errs), '%, max: ', 100*np.max(mse_errs), '%, min: ' ,100* np.min(mse_errs), '%')
    print('----------------------------------------------')
    print('mode of backprojection err:', stats.mode(mse_errs)[0], '%')
    print('----------------------------------------------')
    print('median of backprojection err:', 100*np.median(mse_errs), '%, std: ',np.std(mse_errs))

    plt.close()
    
def results_categorical_histo(pdf_handler, scores_matrix):
    # fig = plt.figure()
    print('scores matrix, top-k accuracy mean across k-folds', np.mean(scores_matrix, axis=0).round(3))
    print('scores matrix, top-k accuracy std across k-folds', np.std(scores_matrix, axis=0).round(3))
    plt.bar(np.arange(27), np.mean(scores_matrix, axis=0), yerr=np.std(scores_matrix, axis=0))
    # plt.show()
    rang = np.arange(0, 28, step=2)
    labels = [str(k) for k in rang ] + ['28',]
    plt.xticks(rang, labels=labels, rotation=45, size=15)
    rang = np.arange(0, 12, step=2)/10
    labels = [str(k) for k in rang ] + ['1.4',]
    plt.yticks(rang, labels=labels, rotation=45, size=15)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.xlabel('K', size=24)
    plt.ylabel('Top-k accuracy', size=24)
    plt.tight_layout(rect=[0, 0, 1, .95])
    # plt.show()
    pdf_handler.savefig()
    pdf_handler.close()