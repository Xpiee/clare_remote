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
from numpy.random import seed
from numpy import array, vstack, hstack, stack
import random as rn
rn.seed(4)
import pickle

import mega_model

from tensorflow.keras import backend as K

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

main_path = r"/home/18ab106/pfiles/clare/Processed_Data_fold"
with open(os.path.join(main_path, 'cola_labels.pickle'), 'rb') as handle:
    sub_label_ecg = pickle.load(handle)

method = 'LOSO'
dataset_name = 'cola'
num_classes = 2

num_combinations = ['ecg', 'eda', 'eeg', 'gze']

for comb in num_combinations:

    print("--------------------------------------------------------------------------")
    print("Training for Type {}, combination {}".format('FF', comb))
    print("--------------------------------------------------------------------------\n")        

    if comb == 'ecg':
        with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)

        in_shape = [(2560, 1)]
        mod_names = ['ecg']

    elif comb == 'eda':
        with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)        
        in_shape = [(2560, 3)]
        mod_names = ['eda']

    elif comb == 'eeg':
        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)        

        in_shape = [(2560, 4)]
        mod_names = ['eeg']            

    elif comb == 'gze':
        with open(os.path.join(main_path, 'cola_gze.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)
        in_shape = [(2560, 2)]
        mod_names = ['gze']

    # make one-fold data here itself.
    mod1, lab1 = support_functions.data_folds.make_data([mod1], sub_label_ecg, num_modality=1)
    mod1, lab1 = unison_shuffled_copies_two(mod1, lab1)

    hs, preds, clr = {}, {}, {}

    path_logs = r'/home/18ab106/pfiles/clare/Data_files/'
    tensorbrd_dir, model_report, model_data, model_score, model_arch, model_fid, model_weights, model_files = create_dirs(path_logs)

    tenFoldSplit = KFold(n_splits=10)
    counter = 1

    for train_index, test_index in tenFoldSplit.split(mod1):

        X1_train, X1_test = mod1[train_index], mod1[test_index]
        y_train, y_test = np.asarray(lab1)[train_index], np.asarray(lab1)[test_index]
        clr[counter], hist, roc_auc, scores, mod_1 = train_model_10fold.training_one_modality(
                                                                                            [X1_train, X1_test],
                                                                                            [y_train, y_test],
                                                                                            counter, tensorbrd_dir,
                                                                                            in_shape, mod_names,
                                                                                            [model_arch, model_weights],
                                                                                            num_classes
                                                                                            )

        with open(os.path.join(model_report, 'Test_fold_{}_report.pickle'.format(counter)), 'wb') as handle:
            pickle.dump(clr, handle, protocol= pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_data, 'Test_fold_{}_data.pickle'.format(counter)), 'wb') as handle:
            pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_score, 'Test_fold_{}_scores.pickle'.format(counter)), 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        create_csv(model_files, clr[counter], method, mod_1, dataset_name=dataset_name)  

        counter += 1   
    
    print("--------------------------------------------------------------------------")
    print('Classfication report for Type {}, Stage {}'.format('ff', comb))    
    score_class(clr)
    print("--------------------------------------------------------------------------")