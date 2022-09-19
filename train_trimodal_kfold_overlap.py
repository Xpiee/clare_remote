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
        tf.config.experimental.set_visible_devices(gpus[5], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
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
from numpy.random import seed
from numpy import array, vstack, hstack, stack
import random as rn
rn.seed(4)
import pickle

import mega_model

from tensorflow.keras import backend as K
import tensorflow.keras.utils as tf_util

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import train_model
import support_functions.data_folds
import train_model_10fold

import warnings
warnings.filterwarnings('ignore')

seed(2)
tf.random.set_seed(42)
print(tf.keras.__version__)

main_path = r"/home/18ab106/pfiles/clare/Processed_Data_fold_overlap"
with open(os.path.join(main_path, 'cola_labels.pickle'), 'rb') as handle:
    sub_label = pickle.load(handle)

method = 'LOSO'
dataset_name = 'cola'
num_classes = 2

# num_combinations = ['ecg', 'eda', 'eeg', 'gze']
num_combinations = ['ecg_eda_eeg', 'ecg_eda_gze', 'ecg_eeg_gze', 'eda_eeg_gze', 'ecg_eda_eeg_gze']
# num_combinations = ['ecg_eeg', 'eda_eeg']

for comb in num_combinations:

    print("--------------------------------------------------------------------------")
    print("Training for Type {}, combination {}".format('FF', comb))
    print("--------------------------------------------------------------------------\n")        

    if comb == 'ecg_eda_eeg':
        with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)
        with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)
        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod3 = pickle.load(handle)
        in_shape = [(2560, 1), (2560, 3), (2560, 4)]
        mod_names = ['ecg', 'eda', 'eeg']

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

    path_logs = r'/home/18ab106/pfiles/clare/Data_files/'
    tensorbrd_dir, model_report, model_data, model_score, model_arch, model_fid, model_weights, model_files = create_dirs(path_logs)

    for fold in mod1.keys():

        X1_train = np.vstack([v_t for idx, v_t in mod1.items() if idx != fold])
        X1_test = mod1[fold]

        X2_train = np.vstack([v_t for idx, v_t in mod2.items() if idx != fold])
        X2_test = mod2[fold]

        X3_train = np.vstack([v_t for idx, v_t in mod3.items() if idx != fold])
        X3_test = mod3[fold]

        if len(mod_names) == 4:
            X4_train = np.vstack([v_t for idx, v_t in mod4.items() if idx != fold])
            X4_test = mod4[fold]

        y_train = [x for idx, y in sub_label.items() for x in y if idx != fold]
        y_test = sub_label[fold]

        y_train = [1 if x > 5 else 0 for x in y_train]
        y_test = [1 if x > 5 else 0 for x in y_test]

        y_train = tf_util.to_categorical(y_train)        
        y_test = tf_util.to_categorical(y_test)        

        if len(mod_names) == 3:

            clr[fold], hist, roc_auc, scores, mod_1 = train_model_10fold.training_three_modality(mod_data = [(X1_train, X2_train, X3_train), (X1_test, X2_test, X3_test)], 
                                                                                                label = [y_train, y_test], i = fold, tensorbrd_dir= tensorbrd_dir,
                                                                                                in_shape = in_shape,
                                                                                                mod_names= mod_names,
                                                                                                save_info = [model_arch, model_weights],
                                                                                                num_classes= num_classes)

        elif len(mod_names) == 4:
            clr[fold], hist, roc_auc, scores, mod_1 = train_model_10fold.training_four_modality(
                                                                                                [(X1_train, X2_train, X3_train, X4_train), (X1_test, X2_test, X3_test, X4_test)],
                                                                                                [y_train, y_test],
                                                                                                fold, tensorbrd_dir,
                                                                                                in_shape, mod_names,
                                                                                                [model_arch, model_weights],
                                                                                                num_classes
                                                                                                )
            

        with open(os.path.join(model_report, 'Test_fold_{}_report.pickle'.format(fold)), 'wb') as handle:
            pickle.dump(clr, handle, protocol= pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_data, 'Test_fold_{}_data.pickle'.format(fold)), 'wb') as handle:
            pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_score, 'Test_fold_{}_scores.pickle'.format(fold)), 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        create_csv(model_files, clr[fold], method, mod_1, dataset_name=dataset_name)  

    print("--------------------------------------------------------------------------")
    print('Classfication report for Type {}, Stage {}'.format('ff', comb))    
    score_class(clr)
    print("--------------------------------------------------------------------------")