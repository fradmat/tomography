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

# def plot_emiss_signal(emiss_mean, emiss_std, post_data_mean, post_data_std, measurement, inf_err_mask,
#                       hps,mll, density_shape,
#                       original_profile):

# def data_heatmap(best_sigmas_grid, sigma_fs, sigma_xs, sigma_errs, pdf_handler):
 

def data_histogram(best_sigmas_grid, sigma_fs, sigma_xs, sigma_errs, fpath1):
    best_sigmas_grid = np.asarray(best_sigmas_grid)
    fig = plt.figure()
    # gs = gridspec.GridSpec(1,2*len(sigma_errs))
    # multi_class = np.zeros(len(sigma_fs) * len(sigma_xs) * len(sigma_errs))
    # print(best_sigmas_grid.shape,len(sigma_fs) * len(sigma_xs) * len(sigma_errs))
    # exit(0)
    num_classes = len(sigma_fs) * len(sigma_xs) * len(sigma_errs)
    multi_class = np.reshape(np.copy(best_sigmas_grid), (best_sigmas_grid.shape[0], num_classes))

    multi_class_cat = np.argmax(multi_class, axis=1)
    # print(multi_class_cat.shape, multi_class_cat[0])
    # exit(0)
    multi_class_sum = pd.Series({'category':multi_class_cat})
    # print(multi_class_sum.shape)
    # exit(0)
    g = sns.countplot(x='category', data=multi_class_sum, color =sns.color_palette("Blues").as_hex()[-2])
    rang = np.arange(0, num_classes+1, step=2)
    labels = [str(k) for k in rang ] #+ ['28',]
    plt.xticks(rang, labels=labels, rotation=45, size=15)
    for tick in g.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    # g.xaxis.set_minor_locator(AutoMinorLocator)
    plt.xlabel('Class', size=24)
    plt.ylabel('Count', size=24)
    plt.title('Distribution of data among classes')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    figure = g.get_figure()
    figure.savefig(fpath1)
    plt.show()
    plt.close()
    

# def plot_measurement(measurement, err, shot, time):
    
# def data_err_histo(mse_errs, fpath1, fpath2, err_type=''):

    
def results_categorical_histo(pdf_handler, scores_matrix, data_type):
    # print('scores matrix, top-k accuracy mean across k-folds', np.mean(scores_matrix, axis=0).round(3))
    # print('scores matrix, top-k accuracy std across k-folds', np.std(scores_matrix, axis=0).round(3))

    num_classes = scores_matrix.shape[1]
    plt.bar(np.arange(num_classes), np.mean(scores_matrix, axis=0), yerr=np.std(scores_matrix, axis=0))
    # plt.show()
    rang = np.arange(0, num_classes + 1, step=2)
    labels = [str(k) for k in rang ] #+ ['28',]
    # print(rang, labels)
    # exit(0)
    plt.xticks(rang, labels=labels, rotation=45, size=15)
    rang = np.arange(0, 12, step=2)/10
    labels = [str(k) for k in rang ] #+ ['1.4',]
    plt.yticks(rang, labels=labels, rotation=45, size=15)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.xlabel('K', size=24)
    plt.ylabel('Top-k accuracy', size=24)
    plt.title(data_type)
    plt.tight_layout(rect=[0, 0, 1, .95])
    plt.show()
    # pdf_handler.savefig()
    pdf_handler.close()