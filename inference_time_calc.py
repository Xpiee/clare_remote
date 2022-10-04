#!/usr/bin/env acii
# with validation set from Training set
import sys
print(sys.executable)

# From Stackoverflow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[4], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
import os
os.environ['PYTHONHASHSEED'] = '0'

from numpy.random import seed
from numpy import array, vstack, hstack, stack
from utils import unison_shuffled_copies, unison_shuffled_copies_two, NDStandardScaler
from utils import mk_dirs, create_csv, create_dirs, f1_m, precision_m, recall_m, create_multicsv
from utils import *

import os
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import seed
from numpy import array, vstack, hstack, stack
import random as rn
rn.seed(4)
import gc
import datetime
import inspect
import pickle
import scipy.io
from scipy import stats
import scipy.signal as scisig
from scipy.stats import zscore
import matplotlib.pyplot as plt

import mega_model
import mega_model_resnet
from tensorflow.keras import backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import class_weight
import h5py
import neurokit2 as nk
from statistics import mean, mode, StatisticsError
from sklearn.preprocessing import MinMaxScaler

import train_model

import warnings
warnings.filterwarnings('ignore')

seed(2)
tf.random.set_seed(42)
print(tf.keras.__version__)

main_path = r"/home/18ab106/pfiles/clare/Processed_Data_2"
with open(os.path.join(main_path, 'cola_labels.pickle'), 'rb') as handle:
    sub_label_ecg = pickle.load(handle)

method = 'LOSO'
dataset_name = 'cola'
num_classes = 2

num_combinations = ['ecg_eda_eeg_gze']

for comb in num_combinations:

    print("--------------------------------------------------------------------------")
    print("Training for Type {}, combination {}".format('FF', comb))
    print("--------------------------------------------------------------------------\n")        

    if comb == 'ecg_gze':
        with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)

        in_shape = [(2560, 1), (2560, 2)]
        mod_names = ['ecg', 'gze']        

    elif comb == 'eda_gze':
        with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)        
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)
        in_shape = [(2560, 3), (2560, 2)]
        mod_names = ['eda', 'gze']
        
    elif comb == 'eeg_gze':
        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)        
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)

        in_shape = [(2560, 4), (2560, 2)]
        mod_names = ['eeg', 'gze']            

    elif comb == 'ecg_eda_gze':
        with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)
        with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)        
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod3 = pickle.load(handle)

        in_shape = [(2560, 1), (2560, 3), (2560, 2)]
        mod_names = ['ecg', 'eda', 'gze']

    elif comb == 'ecg_eeg_gze':
        with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)
        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)        
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod3 = pickle.load(handle)

        in_shape = [(2560, 1), (2560, 4), (2560, 2)]
        mod_names = ['ecg', 'eeg', 'gze']

    elif comb == 'eda_eeg_gze':
        with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)
        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)        
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod3 = pickle.load(handle)

        in_shape = [(2560, 3), (2560, 4), (2560, 2)]
        mod_names = ['eda', 'eeg', 'gze']

    elif comb == 'ecg_eda_eeg_gze':
        with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)        
        with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)
        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod3 = pickle.load(handle)        
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod4 = pickle.load(handle)

        in_shape = [(2560, 1), (2560, 3), (2560, 4), (2560, 2)]
        mod_names = ['ecg', 'eda', 'eeg', 'gze']
        

    hs, preds, clr = {}, {}, {}
    infer_T = {}
    path_logs = r'/home/18ab106/pfiles/clare/Data_files/'
    tensorbrd_dir, model_report, model_data, model_score, model_arch, model_fid, model_weights, model_files = create_dirs(path_logs)

    for i in mod1.keys():

        if i in ['1765']:
            continue

        if len(mod_names) == 2:

            clr[i], hist, roc_auc, scores, mod_1 = train_model.training_binary_modality(mod1, mod2,
                                                                                    sub_label_ecg, i,
                                                                                    tensorbrd_dir, in_shape,
                                                                                    mod_names,
                                                                                    [model_arch, model_weights],
                                                                                    num_classes)

        if len(mod_names) == 3:
            clr[i], hist, roc_auc, scores, mod_1 = train_model.training_three_modality(mod1, mod2, mod3,
                                                                                    sub_label_ecg, i,
                                                                                    tensorbrd_dir, in_shape,
                                                                                    mod_names,
                                                                                    [model_arch, model_weights],
                                                                                    num_classes)

        if len(mod_names) == 4:
            clr[i], hist, roc_auc, scores, mod_1, infer_T[i] = train_model.training_four_modality(mod1, mod2, mod3, mod4,
                                                                                    sub_label_ecg, i,
                                                                                    tensorbrd_dir, in_shape,
                                                                                    mod_names,
                                                                                    [model_arch, model_weights],
                                                                                    num_classes, inference_time=True)


        # with open(os.path.join(model_data, 'Test_fold_{}_data.pickle'.format(i)), 'wb') as handle:
        #     pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(os.path.join(model_score, 'Test_fold_{}_scores.pickle'.format(i)), 'wb') as handle:
        #     pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # create_csv(model_files, clr[i], method, mod_1, dataset_name=dataset_name)            
    
        print("--------------------------------------------------------------------------")
        # print('Classfication report for Type {}, Stage {}'.format('ff', 'ff'))    
        # score_class(clr)
        print('Inference time: {}'.format(infer_T[i]))        
        print("--------------------------------------------------------------------------")

with open(os.path.join(model_report, 'Test_fold_{}_report.pickle'.format(i)), 'wb') as handle:
    pickle.dump(infer_T, handle, protocol= pickle.HIGHEST_PROTOCOL)
    