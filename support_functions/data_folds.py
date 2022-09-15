import os
os.environ['PYTHONHASHSEED'] = '0'

from numpy.random import seed
from numpy import array, vstack, hstack, stack
from utils import unison_shuffled_copies, unison_shuffled_copies_two, NDStandardScaler
from utils import mk_dirs

import os
import numpy as np
import pandas as pd
from numpy.random import seed
from numpy import array, vstack, hstack, stack
import random as rn
rn.seed(4)
import gc

import tensorflow.keras.utils as tf_util

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

def data_combine(X, label):

    X = [vstack(v) for k, v in X.items()]
    y = [hstack(np.asarray(v)) for k, v in label.items()]
    X = vstack(X)

    y = hstack(np.asarray(y))
    y = [1 if x > 5 else 0 for x in y]

    y = tf_util.to_categorical(y)

    return X, y


def make_data(data: list, label: dict, num_modality: int = 1):

    if num_modality == 1:
        mod1 = data[0]
        X, label = data_combine(mod1, label)
        
        return X, label

    elif num_modality == 2:
        mod1 = data[0]
        mod2 = data[1]

        X1, label = data_combine(mod1, label)
        X2, _ = data_combine(mod2, label)
        return X1, X2, label

    elif num_modality == 3:
        mod1 = data[0]
        mod2 = data[1]
        mod3 = data[2]

        X1, label = data_combine(mod1, label)
        X2, _ = data_combine(mod2, label)
        X3, _ = data_combine(mod3, label)

        return X1, X2, X3, label

    elif num_modality == 4:
        mod1 = data[0]
        mod2 = data[1]
        mod3 = data[2]
        mod4 = data[3]

        X1, label = data_combine(mod1, label)
        X2, _ = data_combine(mod2, label)
        X3, _ = data_combine(mod3, label)
        X4, _ = data_combine(mod4, label)

        return X1, X2, X3, X4, label