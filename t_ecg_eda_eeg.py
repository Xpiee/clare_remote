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
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

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

import warnings
warnings.filterwarnings('ignore')


seed(2)
tf.random.set_seed(42)
print(tf.keras.__version__)


main_path = r"/home/18ab106/pfiles/clare/"

# with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
#     sub_dict_ecg = pickle.load(handle)

# with open(os.path.join(main_path, 'cola_labels.pickle'), 'rb') as handle:
#     sub_label_ecg = pickle.load(handle)

# with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
#     sub_dict_eeg = pickle.load(handle)

# with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
#     sub_dict_eda = pickle.load(handle)

with open(os.path.join(main_path, 'cola_labels.pickle'), 'rb') as handle:
    sub_label_ecg = pickle.load(handle)

method = 'LOSO'
dataset_name = 'cola'
num_classes = 2

num_combinations = ['ecg_eeg', 'eda_eeg']

for comb in num_combinations:

    print("--------------------------------------------------------------------------")
    print("Training for Type {}, combination {}".format('FF', comb))
    print("--------------------------------------------------------------------------\n")        


    if comb == 'ecg_eeg':
        with open(os.path.join(main_path, 'cola_ecg.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)

        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)

        in_shape = [(2560, 1), (2560, 4)]
        mod_names = ['ecg', 'eeg']        

    elif comb == 'eda_eeg':
        with open(os.path.join(main_path, 'cola_eda.pickle'), 'rb') as handle:
            mod1 = pickle.load(handle)        

        with open(os.path.join(main_path, 'cola_eeg.pickle'), 'rb') as handle:
            mod2 = pickle.load(handle)

        in_shape = [(2560, 3), (2560, 4)]
        mod_names = ['eda', 'eeg']        

    hs, preds, clr = {}, {}, {}

    path_logs = r'/home/18ab106/pfiles/clare/Data_files/'
    tensorbrd_dir, model_report, model_data, model_score, model_arch, model_fid, model_weights, model_files = create_dirs(path_logs)

    for i in mod1.keys():

        if i in ['1765']:
            continue

        opt = tf.keras.optimizers.Adadelta(learning_rate = 0.0005, rho=0.95)
        tb = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(tensorbrd_dir,
                                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        X_test_1 = mod1[i]
        y_test = sub_label_ecg[i]
        X_test_2 = mod2[i]

        X_test_1 = vstack(X_test_1)
        X_test_2 = vstack(X_test_2)

        y_test = [x for z in y_test for x in z]

        X_1 = [vstack(v) for k, v in mod1.items() if k != i]
        X_2 = [vstack(v) for k, v in mod2.items() if k != i]
        
        y_train = [hstack(np.asarray(v)) for k, v in sub_label_ecg.items() if k != i]

        X_1 = vstack(X_1)
        X_2 = vstack(X_2)

        y_train = hstack(np.asarray(y_train))

        y_train = [1 if x > 5 else 0 for x in y_train]
        y_test = [1 if x > 5 else 0 for x in y_test]
        
        y = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        callbacks_list = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                            patience=20, verbose=1, mode='max', 
                                                            restore_best_weights=True)

        class_wgt = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        wgt = {0:round(class_wgt[0], 2), 1: round(class_wgt[1], 2)}
    #             wgt = {0:round(class_wgt[0], 2), 1: round(class_wgt[1], 2), 2: round(class_wgt[2], 2)}

        model = mega_model.multimodal_classifier(input_shape=in_shape, classes=num_classes, modality_names=mod_names)
        mod_1 = inspect.getsource(mega_model.multimodal_classifier)
        
        model.compile(optimizer=opt, loss=focal_loss_fx(), metrics=['acc', tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average = 'macro')])
        print('Testing on {}'.format(i))

        hist = model.fit([X_1, X_2], y, epochs=300, verbose=2, shuffle=True,
                        batch_size = 256, validation_data = ([X_test_1, X_test_2], y_test),
                        callbacks=[tb, callbacks_list]) # , class_weight=wgt
        y_pred_i = model.predict([X_test_1, X_test_2], batch_size = 128)

        pred_list = list()
        test_y = list()

        for n in range(len(y_pred_i)):
            pred_list.append(np.argmax(y_pred_i[n]))
            test_y.append(np.argmax(y_test[n]))

        gc.collect()

        print(classification_report(pred_list, test_y))
        a = classification_report(pred_list, test_y,
                                    target_names = ['Baseline', 'Stress'],
                                    output_dict=True)

        clr[i] = a
        hs[i] = hist

        roc_auc = roc_auc_score(y_test.astype('int'), y_pred_i, multi_class='ovo', average='weighted')
        scores = {'roc_auc': roc_auc, 'pred_prob': y_pred_i,
                    'pred': pred_list, 'test_cat': y_test, 'test': test_y}

        model.save(os.path.join(model_arch, 'model_{}'.format(i)))
        model_wgt_path = os.path.join(model_weights, '_model_{}'.format(i))
        model.save_weights(os.path.join(model_wgt_path, 'model_{}'.format(i)))

        with open(os.path.join(model_report, 'Test_fold_{}_report.pickle'.format(i)), 'wb') as handle:
            pickle.dump(clr, handle, protocol= pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_data, 'Test_fold_{}_data.pickle'.format(i)), 'wb') as handle:
            pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(model_score, 'Test_fold_{}_scores.pickle'.format(i)), 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        create_csv(model_files, a, method, mod_1, dataset_name=dataset_name)
        K.clear_session()
        
    print("--------------------------------------------------------------------------")
    print('Classfication report for Type {}, Stage {}'.format('ff', 'ff'))    
    score_class(clr)
    print("--------------------------------------------------------------------------")