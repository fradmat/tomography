import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from convnet_sigma_err import *
from keras.utils import plot_model
import random
from data_generator_sigma_err import *
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import accuracy_score
import numpy.ma as ma
import matplotlib as mpl
import seaborn as sns
from plots import plot_emiss_signal, data_err_histo, results_categorical_histo
from gp_real_data import compute_abs_error
import time
from postprocess import post_training
mpl.use('Agg')


 
# dtime = get_date_time_formatted()
exp_id = './gp/' + sys.argv[1]
train_dir = './experiments/' + sys.argv[1]
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
print('Will save this model to', train_dir)

checkpoint_dir = train_dir +'/model_checkpoints/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
logs_dir = train_dir +'/logs/'
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)
    
hp_dic = load_dic(exp_id + '/hp_params')
sigma_fs = hp_dic['sigma_fs']
sigma_xs = hp_dic['sigma_xs']
sigma_errs = hp_dic['sigma_errs']   
    
epoch_size = 128
no_epocs = 50
no_sensors = 208

bsize = len(sigma_fs)*len(sigma_xs)*len(sigma_errs)*8 #32 samples per hyperparameter (of which there are 3*3*3)
shots = get_shots()
train_val_shots = shots#[:-11]
# holdout_shots = shots[-11:]
ensemble_size = 10

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
# -------- Training --------
params_random = {'dim': no_sensors,
                  'no_batches_ep': epoch_size, 
                  'shuffle': True,
                  'exp_id': exp_id,
                  'batch_size': bsize,
                  'epoch_size':epoch_size,
                  'shots': train_val_shots,
                  'n_splits':ensemble_size
                  }

print('experiment parameters', params_random)


mainGenerator = DataGenerator(**params_random)
ensemble = []
pred_hps_top_all = []
scores_matrix = {'train':np.empty((ensemble_size, len(sigma_fs)*len(sigma_errs)*len(sigma_xs))),
                 'test':np.empty((ensemble_size, len(sigma_fs)*len(sigma_errs)*len(sigma_xs)))}
val_ms = []
val_errs = []
used_ms_all = []
st_all = []
pred_deltas = []
for k_fold in range(mainGenerator.n_splits):
    print('----------------------------------------------Training member', k_fold+1, 'of ensemble...----------------------------------------------')
    cnn = model(no_sensors, len(sigma_fs), len(sigma_xs), len(sigma_errs), k_fold)
    cnn.compile(loss={'single_mclass':'categorical_crossentropy'},
                optimizer='adam',
                metrics={'single_mclass':['categorical_accuracy', top_2_accuracy, top_3_accuracy]})
    
    train_generator = TrainDataGenerator(k_fold, mainGenerator)
    train_gen = next(iter(train_generator))
    val_generator = ValDataGenerator(k_fold, mainGenerator)
    val_gen = next(iter(val_generator))

    train_start = time.time()
    cnn.fit_generator(generator = train_gen, steps_per_epoch=epoch_size, epochs=no_epocs, validation_data=val_gen, validation_steps=epoch_size)#, callbacks=[saveCheckpoint,]) #,tb ,, validation_steps=bsize
    train_end = time.time()
    print('----------------------------------------------')
    print('training took:', train_end-train_start, 'seconds')
    print('----------------------------------------------')
    inputs, targets, control = val_generator.get_all_items()
    measurements = inputs['m']#[:10]
    errs = inputs['m_err']#[:10]
    st = control['s&t']
    # print(np.sum(errs, axis=1))
    used_ms = np.sum(errs, axis=1)/208
    used_ms_all.extend(used_ms)
    st_all.extend(st)
    # print(np.asarray(st_all).shape)
    # print(used_ms.shape)
    # # print(used_ms)
    # exit(0)
    labels = targets['single_mclass']#[:10]
    labels_argmax = np.argmax(labels, axis=1)
    val_ms.extend(measurements)
    val_errs.extend(errs)
    
    print('classifier predicting on its val data, whose shape is', measurements.shape, errs.shape, labels.shape, labels_argmax.shape)#, train_val_hps[:100].shape)
    pred_start = time.time()
    pred_hps = cnn.predict([measurements, errs])
    pred_end = time.time()
    pred_delta = pred_end-pred_start
    print('----------------------------------------------')
    print('prediction took:', pred_delta, 'seconds')
    pred_deltas.append(pred_delta)
    print('----------------------------------------------')
    # pred_hps = np.zeros((len(labels),27))
    # arr = np.zeros(27)
    # arr[21] = 1
    # pred_hps[0] = arr
    argsort_preds = np.argsort(pred_hps, axis=1)
    pred_hps_top_all.extend(argsort_preds[:, -1])
    print('shape of predictions of whole dataset, ', pred_hps.shape)
    # print('argsort_ensemble_preds', argsort_preds.shape, train_val_hps[:100].shape, argsort_preds)
    # print(pred_hps_all[0], argsort_preds[0])
    # exit(0)
    acc_sum = 0
    acc_list = []
    acc_h = []
    for k in range(len(sigma_fs)*len(sigma_errs)*len(sigma_xs)):
        acc_int = accuracy_score(labels_argmax, argsort_preds[:,-1-k], normalize=False)
        acc_sum += acc_int
        acc_h.append(acc_sum/len(labels_argmax))
        acc_list.extend(np.ones(acc_sum) * k)
        # print('hps acc., top ' + str(k+1) + ':', acc_int, 'out of', len(labels_argmax), 'total:', acc_sum, 'cumulative fraction=', acc_sum/len(labels_argmax))
    # ensemble_top_preds = argsort_preds[:,-1]
    # np.save(exp_id + '/pred_hps', ensemble_top_preds)
    scores_matrix[str(val_generator)][k_fold] = np.asarray(acc_h)
    
print('----------------------------------------------')
print('average fraction of used measurements per data point:', np.mean(used_ms_all))
print('----------------------------------------------')
print('time the classifiers took to predict across their k-fold split:', np.asarray(pred_deltas))
print('average time the classifiers took to predict across their k-fold split:', np.mean(pred_deltas))
print('total time the classifiers took to predict across the entire k-fold split:', np.sum(pred_deltas))
print('----------------------------------------------')

np.save(exp_id + '/scores_matrix', scores_matrix[str(val_generator)])
# print(scores_matrix[str(val_generator)])
# print(scores_matrix[str(val_generator)].shape)
# print(np.mean(scores_matrix[str(val_generator)], axis=0))
# print(np.var(scores_matrix[str(val_generator)], axis=0))
pdf_handler = PdfPages(exp_id + '/accuracy_histogram.pdf')
results_categorical_histo(pdf_handler, scores_matrix[str(val_generator)])

pred_hps_top_all = np.asarray(pred_hps_top_all)
val_ms = np.asarray(val_ms)
val_errs = np.asarray(val_errs)
print('shape of all validation predictions, measurements and errs:', pred_hps_top_all.shape, val_ms.shape, val_errs.shape)
np.save(exp_id + '/val_ms', val_ms)
np.save(exp_id + '/val_errs', val_errs)
np.save(exp_id + '/val_sts', st_all)
np.save(exp_id + '/pred_hps_top_all_kfolds', pred_hps_top_all)
post_training(str(val_generator) + '_histogram.pdf', exp_id, hp_dic, pred_hps_top_all, val_ms, val_errs, st_all)


# post_training('train_val_histogram.pdf', exp_id,  ensemble, train_val_measurements, train_val_m_errs, train_val_hps, train_val_hps_cat, hp_dic)
# post_training('holdout_histogram.pdf', exp_id, ensemble, holdout_measurements, holdout_m_errs, holdout_hps, holdout_hps_cat, hp_dic)



# 
# ensemble_top_preds = np.load(exp_id + '/pred_hps.npy')
# print(ensemble_top_preds.shape)
# print(ensemble_top_preds[:5])
# # ensemble_top_preds = np.arange(27)
# # ensemble_top_preds = np.asarray([0,9,18,1,10,19,2,11,20,3,12,21,4,13,22,5,14,23,6,15,24,7,16,25,8,17,26])
# print(ensemble_top_preds)
# pred_errs = np.floor_divide(ensemble_top_preds, 9)
# pred_xs = np.remainder(ensemble_top_preds, 3)
# pred_fs = ((ensemble_top_preds-(9* pred_errs+pred_xs ))/3).astype(int)
# print(pred_errs)
# print(pred_xs)
# print(pred_fs)
# 
# for k in range(len(ensemble_top_preds)):
#     best_sigma_grid = np.zeros(len(sigma_fs)*len(sigma_xs)*len(sigma_errs)).reshape(len(sigma_errs), len(sigma_fs), len(sigma_xs))
#     best_sigma_grid[pred_errs[k], pred_fs[k], pred_xs[k]] = 1
#     print(pred_errs[k], pred_xs[k], pred_fs[k], np.argmax(best_sigma_grid.flatten()), best_sigma_grid.flatten())
#     print()