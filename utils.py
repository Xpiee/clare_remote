import numpy as np
import pandas as pd
from numpy import array, vstack, hstack, stack
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import pickle
# import h5py
import neurokit2 as nk
import scipy.io
from scipy import stats
import scipy.signal as scisig
from scipy.stats import zscore
from scipy.signal import butter, lfilter, filtfilt, welch
import scipy.misc
import datetime
import os
import tensorflow as tf

def recall_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+k.epsilon()))

def create_dirs(path):
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_files = "experiment_" + time_stamp
    log_dir = path + model_files
    tensorbrd_dir = os.path.join(log_dir, "t_b")
    tb_files = os.path.join(tensorbrd_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    model_rp_dir = os.path.join(log_dir, "reports")

    model_report = os.path.join(model_rp_dir, "report")
    model_data = os.path.join(model_rp_dir, "data")
    model_score = os.path.join(model_rp_dir, "score")
    model_arch = os.path.join(log_dir, "model_arch")
    model_fid = os.path.join(model_rp_dir, "fid")
    model_weights = os.path.join(log_dir, "model_weights")

    print("Model files saved in: ", log_dir)
    print("Tensorboard files for model saved in: ", tensorbrd_dir)
    print("Model report files saved in : ", model_rp_dir)
    print("Model weights saved in : ", model_weights)

    dirs = [tensorbrd_dir, model_report, model_data, model_score, model_arch, model_fid, model_weights]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    dirs = [tensorbrd_dir, model_report, model_data, model_score, model_arch, model_fid, model_weights, model_files]
    return dirs


def create_csv(exp_num, cls_rprt, method, model_fnc, dataset_name):
    df = pd.DataFrame(cls_rprt)
    df1 = df.drop(columns=['accuracy'])
    df1 = df1.iloc[:3, :]
    df2 = df['accuracy'][0]
    df_list = df1.to_numpy().tolist()
    df_list = [item for sublist in df_list for item in sublist]
    cols = ['precision_nstrs', 'precision_strs', 'precision_mcavg', 'precision_wtavg',
            'recall_nstrs', 'recall_strs', 'recall_mcavg', 'recall_wtavg',
            'f1_nstrs', 'f1_strs', 'f1_mcavg', 'f1_wtavg']
    df_cls = pd.DataFrame(columns=cols)
    df_cls.loc[len(df_cls)] = df_list
    df_cls['acc'] = df2
    df_cls['experiment'] = exp_num
    df_cls['method'] = method
    df_cls['model'] = model_fnc
    df_cls_1 = df_cls[['experiment', 'method', 'model', 'acc', 'f1_nstrs', 'f1_strs', 'f1_mcavg', 'f1_wtavg',
                    'precision_nstrs', 'precision_strs', 'precision_mcavg', 'precision_wtavg',
                    'recall_nstrs', 'recall_strs', 'recall_mcavg', 'recall_wtavg']].copy()
    df_cls_1.to_csv('{}_results.csv'.format(dataset_name), mode='a', header=False, index=False)
    
def create_multicsv(exp_num, cls_rprt, method, model_fnc, dataset_name):
    df = pd.DataFrame(cls_rprt)
    df1 = df.drop(columns=['accuracy'])
    df1 = df1.iloc[:3, :]
    df2 = df['accuracy'][0]
    df_list = df1.to_numpy().tolist()
    df_list = [item for sublist in df_list for item in sublist]
    cols = ['precision_base', 'precision_amse', 'precision_strs', 'precision_mcavg', 'precision_wtavg',
            'recall_base', 'recall_amse', 'recall_strs', 'recall_mcavg', 'recall_wtavg',
            'f1_base', 'f1_amse', 'f1_strs', 'f1_mcavg', 'f1_wtavg']
    df_cls = pd.DataFrame(columns=cols)
    df_cls.loc[len(df_cls)] = df_list
    df_cls['acc'] = df2
    df_cls['experiment'] = exp_num
    df_cls['method'] = method
    df_cls['model'] = model_fnc
    df_cls_1 = df_cls[['experiment', 'method', 'model', 'acc', 'f1_base', 'f1_amse', 'f1_strs', 'f1_mcavg', 'f1_wtavg',
                    'precision_base', 'precision_amse', 'precision_strs', 'precision_mcavg', 'precision_wtavg',
                    'recall_base', 'recall_amse', 'recall_strs', 'recall_mcavg', 'recall_wtavg']].copy()
    df_cls_1.to_csv('{}_results.csv'.format(dataset_name), mode='a', header=False, index=False)

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def unison_shuffled_copies_two(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def mk_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X



class NDMinMaxScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = MinMaxScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

# scale an array of images to a new size
from skimage.transform import resize
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

def feat_rep_extract(model, model_layer, inputs_list):
    keras_fcn = K.function([model.input], [model_layer.output])
    feat_rep = keras_fcn([inputs_list, 1])[0]
    return feat_rep

# Define data load for k folds functions below:

def kload_wesad(basic_path):

    with open(os.path.join(basic_path, "wesad_ecg_samples_kfold_10_overlap_60.pickle"), 'rb') as handle:
        sub_dict_ecg = pickle.load(handle)
    with open(os.path.join(basic_path, "wesad_eda_samples_kfold_10_overlap_60.pickle"), 'rb') as handle:
        sub_dict_eda = pickle.load(handle)
    with open(os.path.join(basic_path, "wesad_ecg_labels_kfold_10_overlap_60.pickle"), 'rb') as handle:
        sub_label_eda = pickle.load(handle)
    return sub_dict_eda, sub_dict_ecg, sub_label_eda

def kload_swell_stress(basic_path):

    with open(os.path.join(basic_path, "kfolds_swell_ecg_2.pickle"), 'rb') as handle:
        sub_dict_ecg = pickle.load(handle)
    with open(os.path.join(basic_path, "kfolds_swell_eda_2.pickle"), 'rb') as handle:
        sub_dict_eda = pickle.load(handle)
    with open(os.path.join(basic_path, "kfolds_swell_labels_2.pickle"), 'rb') as handle:
        sub_label_eda = pickle.load(handle)
    return sub_dict_eda, sub_dict_ecg, sub_label_eda

def kload_swell_valence(basic_path):

    with open(os.path.join(basic_path, "kfolds_swell_ecg_2.pickle"), 'rb') as handle:
        sub_dict_ecg = pickle.load(handle)
    with open(os.path.join(basic_path, "kfolds_swell_eda_2.pickle"), 'rb') as handle:
        sub_dict_eda = pickle.load(handle)
    with open(os.path.join(basic_path, "kfolds_swell_labels_va_2.pickle"), 'rb') as handle:
        sub_label_eda = pickle.load(handle)
    return sub_dict_eda, sub_dict_ecg, sub_label_eda

def kload_swell_arousal(basic_path):

    with open(os.path.join(basic_path, "kfolds_swell_ecg_2.pickle"), 'rb') as handle:
        sub_dict_ecg = pickle.load(handle)
    with open(os.path.join(basic_path, "kfolds_swell_eda_2.pickle"), 'rb') as handle:
        sub_dict_eda = pickle.load(handle)
    with open(os.path.join(basic_path, "kfolds_swell_labels_ar_2.pickle"), 'rb') as handle:
        sub_label_eda = pickle.load(handle)
    return sub_dict_eda, sub_dict_ecg, sub_label_eda


def kload_data(dataset):
    if dataset == 'Wesad Stress':
        basic_path = r"D:/Work/Artificial Intelligence/Masters_Project/Datasets and Code/WESAD/Pickle/kfold/v5"
        sub_dict_eda, sub_dict_ecg, sub_label_eda = load_wesad(basic_path)
    elif dataset == 'Swell Arousal':
        basic_path = r"D:/Work/Artificial Intelligence/Masters_Project/Datasets and Code/SWELL/Pickle/kfolds"
        sub_dict_eda, sub_dict_ecg, sub_label_eda = load_swell_arousal(basic_path)
    elif dataset == 'Swell Valence':
        basic_path = r"D:/Work/Artificial Intelligence/Masters_Project/Datasets and Code/SWELL/Pickle/kfolds"
        sub_dict_eda, sub_dict_ecg, sub_label_eda = load_swell_valence(basic_path)
    elif dataset == 'Swell Stress':
        basic_path = r"D:/Work/Artificial Intelligence/Masters_Project/Datasets and Code/SWELL/Pickle/kfolds"        
        sub_dict_eda, sub_dict_ecg, sub_label_eda = load_swell_stress(basic_path)
    return sub_dict_eda, sub_dict_ecg, sub_label_eda


import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class NDMinMaxScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = MinMaxScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X
    
def focal_loss_fx(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

def score_class(dict_1):
    acc = 0
    base_f1 = 0
    stress_f1 = 0
    macro_f1 = 0
    wgt_f1 = 0
    for i in dict_1.keys():
        acc = acc + dict_1[i]['accuracy']
        base_f1 = base_f1 + dict_1[i]['Baseline']['f1-score']
        stress_f1 = stress_f1 + dict_1[i]['Stress']['f1-score']
        macro_f1 = macro_f1 + dict_1[i]['macro avg']['f1-score']
        wgt_f1 = wgt_f1 + dict_1[i]['weighted avg']['f1-score']

    print("Average Accuracy: ", acc/len(dict_1))
    print("F1 score for Baseline: ", base_f1/len(dict_1))
    print("F1 score for Stress: ", stress_f1/len(dict_1))
    print("Macro F1: ", macro_f1/len(dict_1))
    print("Weighted F1: ", wgt_f1/len(dict_1))
    
def score_multiclass(dict_1):
    acc = 0
    base_f1 = 0
    amuse_f1 = 0
    stress_f1 = 0
    macro_f1 = 0
    wgt_f1 = 0
    for i in dict_1.keys():
        acc = acc + dict_1[i]['accuracy']
        base_f1 = base_f1 + dict_1[i]['Baseline']['f1-score']
        amuse_f1 = amuse_f1 + dict_1[i]['Amusement']['f1-score']
        stress_f1 = stress_f1 + dict_1[i]['Stress']['f1-score']
        macro_f1 = macro_f1 + dict_1[i]['macro avg']['f1-score']
        wgt_f1 = wgt_f1 + dict_1[i]['weighted avg']['f1-score']

    print("Average Accuracy: ", acc/len(dict_1))
    print("F1 score for Baseline: ", base_f1/len(dict_1))
    print("F1 score for Amusement: ", amuse_f1/len(dict_1))
    print("F1 score for Stress: ", stress_f1/len(dict_1))
    print("Macro F1: ", macro_f1/len(dict_1))
    print("Weighted F1: ", wgt_f1/len(dict_1))