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

import mega_model_kfold
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
from sklearn.model_selection import KFold

import h5py
import neurokit2 as nk
from statistics import mean, mode, StatisticsError
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


seed(2)
tf.random.set_seed(42)
print(tf.keras.__version__)

def training_one_modality(mod1: list, label: list, i: int,
                            tensorbrd_dir, in_shape, mod_names, save_info, num_classes):

    model_arch, model_weights = save_info
    X_train, X_test = mod1
    y_train, y_test = label

    opt = tf.keras.optimizers.Adamax(learning_rate = 0.0005)
    tb = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(tensorbrd_dir,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    callbacks_list = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                        patience=30, verbose=1, mode='max', 
                                                        restore_best_weights=True)

    # if mod_names[0] == 'gze':
    #     model = mega_model.gze_arch(in_shape,
    #                                 mod_name=mod_names[0],
    #                                 classes=num_classes, 
    #                                 is_unimodal=True)

    if mod_names[0] == 'eeg':
        model = mega_model_kfold.eeg_arch(in_shape,
                                    mod_name=mod_names[0],
                                    classes=num_classes, 
                                    is_unimodal=True)
        mod_1 = inspect.getsource(mega_model_kfold.eeg_arch)

    else:
        model = mega_model_kfold.unimodal_Kfold(in_shape,
                                    mod_name=mod_names[0],
                                    classes=num_classes, 
                                    is_unimodal=True)
        mod_1 = inspect.getsource(mega_model_kfold.unimodal_Kfold)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                    metrics=['acc', tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average = 'macro')])
    
    print('Testing on {}'.format(i))

    hist = model.fit([X_train], y_train, epochs=300, verbose=2, shuffle=True,
                    batch_size = 512, validation_data = ([X_test], y_test),
                    callbacks=[tb, callbacks_list]) # , class_weight=wgt
    y_pred_i = model.predict([X_test], batch_size = 128)

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

    roc_auc = roc_auc_score(y_test.astype('int'), y_pred_i, multi_class='ovo', average='weighted')
    scores = {'roc_auc': roc_auc, 'pred_prob': y_pred_i,
                'pred': pred_list, 'test_cat': y_test, 'test': test_y}

    model.save(os.path.join(model_arch, 'model_{}'.format(i)))
    model_wgt_path = os.path.join(model_weights, '_model_{}'.format(i))
    model.save_weights(os.path.join(model_wgt_path, 'model_{}'.format(i)))

    return a, hist, roc_auc, scores, mod_1

def training_two_modality(mod_data: list, label: list, i: int,
                            tensorbrd_dir, in_shape, mod_names, save_info, num_classes):

    model_arch, model_weights = save_info
    X1_train, X2_train = mod_data[0]
    X1_test, X2_test = mod_data[1]
    y_train, y_test = label

    opt = tf.keras.optimizers.Adamax(learning_rate = 0.0005)
    tb = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(tensorbrd_dir,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    callbacks_list = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                        patience=30, verbose=1, mode='max', 
                                                        restore_best_weights=True)

    model = mega_model_kfold.bimodal_Kfold(in_shape,
                                mod_name=mod_names,
                                classes=num_classes, 
                                is_unimodal=False)
    mod_1 = inspect.getsource(mega_model_kfold.bimodal_Kfold)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['acc', tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average = 'macro')])
    
    print('Testing on {}'.format(i))

    hist = model.fit([X1_train, X2_train], y_train, epochs=300, verbose=2, shuffle=True,
                    batch_size = 512, validation_data = ([X1_test, X2_test], y_test),
                    callbacks=[tb, callbacks_list]) # , class_weight=wgt
    y_pred_i = model.predict([X1_test, X2_test], batch_size = 128)

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

    roc_auc = roc_auc_score(y_test.astype('int'), y_pred_i, multi_class='ovo', average='weighted')
    scores = {'roc_auc': roc_auc, 'pred_prob': y_pred_i,
                'pred': pred_list, 'test_cat': y_test, 'test': test_y}

    model.save(os.path.join(model_arch, 'model_{}'.format(i)))
    model_wgt_path = os.path.join(model_weights, '_model_{}'.format(i))
    model.save_weights(os.path.join(model_wgt_path, 'model_{}'.format(i)))

    return a, hist, roc_auc, scores, mod_1


def training_binary_modality(mod1, mod2, sub_label_ecg, i, tensorbrd_dir, in_shape, mod_names, save_info, num_classes):

    model_arch, model_weights = save_info

    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
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
                                                        patience=30, verbose=1, mode='max', 
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

    roc_auc = roc_auc_score(y_test.astype('int'), y_pred_i, multi_class='ovo', average='weighted')
    scores = {'roc_auc': roc_auc, 'pred_prob': y_pred_i,
                'pred': pred_list, 'test_cat': y_test, 'test': test_y}


    # clr[i] = a
    # hs[i] = hist

    model.save(os.path.join(model_arch, 'model_{}'.format(i)))
    model_wgt_path = os.path.join(model_weights, '_model_{}'.format(i))
    model.save_weights(os.path.join(model_wgt_path, 'model_{}'.format(i)))

    return a, hist, roc_auc, scores, mod_1


def training_three_modality(mod1, mod2, mod3, sub_label_ecg, i, tensorbrd_dir, in_shape, mod_names, save_info, num_classes):

    model_arch, model_weights = save_info

    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    tb = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(tensorbrd_dir,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    X_test_1 = mod1[i]
    y_test = sub_label_ecg[i]
    X_test_2 = mod2[i]
    X_test_3 = mod3[i]

    X_test_1 = vstack(X_test_1)
    X_test_2 = vstack(X_test_2)
    X_test_3 = vstack(X_test_3)

    y_test = [x for z in y_test for x in z]

    X_1 = [vstack(v) for k, v in mod1.items() if k != i]
    X_2 = [vstack(v) for k, v in mod2.items() if k != i]
    X_3 = [vstack(v) for k, v in mod3.items() if k != i]
    
    y_train = [hstack(np.asarray(v)) for k, v in sub_label_ecg.items() if k != i]

    X_1 = vstack(X_1)
    X_2 = vstack(X_2)
    X_3 = vstack(X_3)

    y_train = hstack(np.asarray(y_train))

    y_train = [1 if x > 5 else 0 for x in y_train]
    y_test = [1 if x > 5 else 0 for x in y_test]
    
    y = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    callbacks_list = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                        patience=30, verbose=1, mode='max', 
                                                        restore_best_weights=True)

    class_wgt = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    wgt = {0:round(class_wgt[0], 2), 1: round(class_wgt[1], 2)}
#             wgt = {0:round(class_wgt[0], 2), 1: round(class_wgt[1], 2), 2: round(class_wgt[2], 2)}

    model = mega_model.multimodal_classifier(input_shape=in_shape, classes=num_classes, modality_names=mod_names)
    mod_1 = inspect.getsource(mega_model.multimodal_classifier)
    
    model.compile(optimizer=opt, loss=focal_loss_fx(), metrics=['acc', tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average = 'macro')])
    print('Testing on {}'.format(i))

    hist = model.fit([X_1, X_2, X_3], y, epochs=300, verbose=2, shuffle=True,
                    batch_size = 256, validation_data = ([X_test_1, X_test_2, X_test_3], y_test),
                    callbacks=[tb, callbacks_list]) # , class_weight=wgt
    y_pred_i = model.predict([X_test_1, X_test_2, X_test_3], batch_size = 128)

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

    roc_auc = roc_auc_score(y_test.astype('int'), y_pred_i, multi_class='ovo', average='weighted')
    scores = {'roc_auc': roc_auc, 'pred_prob': y_pred_i,
                'pred': pred_list, 'test_cat': y_test, 'test': test_y}


    # clr[i] = a
    # hs[i] = hist

    model.save(os.path.join(model_arch, 'model_{}'.format(i)))
    model_wgt_path = os.path.join(model_weights, '_model_{}'.format(i))
    model.save_weights(os.path.join(model_wgt_path, 'model_{}'.format(i)))

    return a, hist, roc_auc, scores, mod_1


def training_four_modality(mod1, mod2, mod3, mod4, sub_label_ecg, i, tensorbrd_dir, in_shape, mod_names, save_info, num_classes):

    model_arch, model_weights = save_info

    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    tb = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(tensorbrd_dir,
                                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    X_test_1 = mod1[i]
    y_test = sub_label_ecg[i]
    X_test_2 = mod2[i]
    X_test_3 = mod3[i]
    X_test_4 = mod4[i]

    X_test_1 = vstack(X_test_1)
    X_test_2 = vstack(X_test_2)
    X_test_3 = vstack(X_test_3)
    X_test_4 = vstack(X_test_4)

    y_test = [x for z in y_test for x in z]

    X_1 = [vstack(v) for k, v in mod1.items() if k != i]
    X_2 = [vstack(v) for k, v in mod2.items() if k != i]
    X_3 = [vstack(v) for k, v in mod3.items() if k != i]
    X_4 = [vstack(v) for k, v in mod4.items() if k != i]
    
    y_train = [hstack(np.asarray(v)) for k, v in sub_label_ecg.items() if k != i]

    X_1 = vstack(X_1)
    X_2 = vstack(X_2)
    X_3 = vstack(X_3)
    X_4 = vstack(X_4)

    y_train = hstack(np.asarray(y_train))

    y_train = [1 if x > 5 else 0 for x in y_train]
    y_test = [1 if x > 5 else 0 for x in y_test]
    
    y = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    callbacks_list = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                                        patience=30, verbose=1, mode='max', 
                                                        restore_best_weights=True)

    class_wgt = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    wgt = {0:round(class_wgt[0], 2), 1: round(class_wgt[1], 2)}
#             wgt = {0:round(class_wgt[0], 2), 1: round(class_wgt[1], 2), 2: round(class_wgt[2], 2)}

    model = mega_model.multimodal_classifier(input_shape=in_shape, classes=num_classes, modality_names=mod_names)
    mod_1 = inspect.getsource(mega_model.multimodal_classifier)
    
    model.compile(optimizer=opt, loss=focal_loss_fx(), metrics=['acc', tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average = 'macro')])
    print('Testing on {}'.format(i))

    hist = model.fit([X_1, X_2, X_3, X_4], y, epochs=300, verbose=2, shuffle=True,
                    batch_size = 256, validation_data = ([X_test_1, X_test_2, X_test_3, X_test_4], y_test),
                    callbacks=[tb, callbacks_list]) # , class_weight=wgt
    y_pred_i = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size = 128)

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

    roc_auc = roc_auc_score(y_test.astype('int'), y_pred_i, multi_class='ovo', average='weighted')
    scores = {'roc_auc': roc_auc, 'pred_prob': y_pred_i,
                'pred': pred_list, 'test_cat': y_test, 'test': test_y}

    # clr[i] = a
    # hs[i] = hist

    model.save(os.path.join(model_arch, 'model_{}'.format(i)))
    model_wgt_path = os.path.join(model_weights, '_model_{}'.format(i))
    model.save_weights(os.path.join(model_wgt_path, 'model_{}'.format(i)))

    return a, hist, roc_auc, scores, mod_1