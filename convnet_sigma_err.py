import numpy as np
from keras.layers import Conv1D, Input, MaxPooling1D, Flatten, concatenate, Dense, Dropout, BatchNormalization, Reshape, Activation, ELU, Lambda
from keras.models import Model, load_model
from keras.callbacks import Callback, TensorBoard
import keras.backend as K
from keras.activations import softmax

from keras.losses import categorical_crossentropy
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import sys
# from matplotlib.mlab import griddata
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import scipy.ndimage as ndimage
import pickle
import sys
from data_generator import *
from gaussian_process import *
import tensorflow as tf

def model(no_sensors, no_sigmafs, no_sigmaxs, no_sigmaerrs, k_fold):
    conv_input = Input(shape=(no_sensors,1), dtype='float32', name='m')
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(conv_input) #input len = 208
    x_conv = Conv1D(32, 3, activation='relu', padding='same')(x_conv) #input len = 208
    # x_conv = Dropout(.25)(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    # x_conv = Dropout(.25)(x_conv)
    x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv) #input len = 104
    x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv) #input len = 104
    x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv) #input len = 104
    x_conv = BatchNormalization()(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    # x_conv = BatchNormalization()(x_conv)
    # x_conv = Dropout(.25)(x_conv)
    x_conv = Conv1D(128, 3, activation='relu', padding='same')(x_conv) #input len = 52
    x_conv = Conv1D(128, 3, activation='relu', padding='same')(x_conv) #input len = 52
    x_conv = Conv1D(128, 3, activation='relu', padding='same')(x_conv) #input len = 52
    x_conv = BatchNormalization()(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    # x_conv = Dropout(.25)(x_conv)
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv) #input len = 26
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv) #input len = 26
    x_conv = Conv1D(256, 3, activation='relu', padding='same')(x_conv) #input len = 26
    x_conv = BatchNormalization()(x_conv)
    x_conv = MaxPooling1D(2)(x_conv)
    # x_conv = Dropout(.25)(x_conv)
    # x_conv = Conv1D(512, 5, activation='relu', padding='same')(x_conv) #input len = 13
    # x_conv = Conv1D(512, 5, activation='relu', padding='same')(x_conv) #input len = 13
    # x_conv = Conv1D(512, 5, activation='relu', padding='same')(x_conv) #input len = 13
    # x_conv = MaxPooling1D(2, strides=2)(x_conv)
    # 
    conv_out = Flatten()(x_conv)

    # full_c_input = Input(shape=(no_sensors,), dtype='float32', name='m_err')
    # 
    # dense_out = Dense(2048, activation='relu')(concatenate([conv_out, full_c_input])) #concatenate([conv_out, full_c_input])
    dense_out = Dense(2048, activation='relu')(conv_out) 
    dense_out = BatchNormalization()(dense_out)
    # dense_out = Dropout(.5)(dense_out)
    
    
    target = Dense(2048, activation='relu')(dense_out)
    target = BatchNormalization()(target)
    # target = Dropout(.5)(target)
    # target = Dense(512, activation='relu')(target)
    target = Dense(512, activation='relu')(target)
    target = BatchNormalization()(target)
    target = Dropout(.5)(target)
    target_single_mclass = Dense(no_sigmafs * no_sigmaxs * no_sigmaerrs, activation='softmax', name='single_mclass')(target)
    
    # model = Model(inputs=[conv_input], outputs=[target_f, target_x, target_err]) #target_phi, target_rho, target_f, target_errs, target_true
    
    model = Model(inputs=[conv_input,full_c_input], outputs=[target_single_mclass,]) #full_c_input
    
    if k_fold == 0:
        print(model.summary())
    return model