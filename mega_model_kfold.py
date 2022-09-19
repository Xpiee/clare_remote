import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Dense
from tensorflow.keras.layers import Input, Flatten, concatenate
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
k.set_image_data_format('channels_last')
k.set_learning_phase(1)
import tensorflow

def conv_blk1D(X_eda_ip, filters, f, strides, stage, stype, l2_cnn):
    F1, F2 = filters
    f1, f2 = f
    s1, s2 = strides
    conv_name = 'conv_' + str(stage) + "_" + str(stype)
    batch_name = 'bn_' + str(stage) + "_" + str(stype)
    act_name = 'act_' + str(stage) + "_" + str(stype)
    maxpool_name = 'mp_' + str(stage) + "_" + str(stype)
    
    X = Conv1D(F1, f1, strides=s1, name= conv_name + 'a',
                   kernel_regularizer = l2_cnn, padding = 'same',
                   kernel_initializer=glorot_uniform(seed=0))(X_eda_ip)
#     X = BatchNormalization(name=batch_name + 'a')(X)
    X = Activation('relu', name=act_name + 'a')(X)

    X = Conv1D(F2, f2, strides=s2, padding = 'same', name= conv_name + 'b',
                   kernel_regularizer=l2_cnn,
                   kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(name=batch_name + 'b')(X)
    X = Activation('relu', name=act_name + 'b')(X)
    X = MaxPooling1D(2, strides=2, name=maxpool_name)(X)
    return X

def unimodal(X_in_shpe, mod_name='ecg', l2_dense = 0.01, l2_cnn = 0.01, glrt = glorot_uniform(seed=4), classes=2, is_unimodal=True):

    if is_unimodal:
        X_in = Input(X_in_shpe[0])
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

    else: X_in = X_in_shpe

    X = conv_blk1D(X_in, [32, 32], [64, 64], [1, 3], 'stage1', mod_name, l2_cnn)
    X = conv_blk1D(X, [64, 64], [32, 32], [1, 3], 'stage2', mod_name, l2_cnn)
    X = conv_blk1D(X, [128, 128], [17, 17], [1, 3], 'stage3', mod_name, l2_cnn)
    X = conv_blk1D(X, [256, 256], [7, 7], [1, 3], 'stage4', mod_name, l2_cnn)

    flatten_X = Flatten()(X)

    dense_X = Dense(512, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense)(flatten_X)
    
    dense_X= Dense(128, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense, name=mod_name + "bf_merge")(dense_X)

    if is_unimodal:
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(dense_X)
        model = Model(inputs = X_in, outputs = out)
        return model

    return dense_X

def unimodal_Kfold(X_in_shpe, mod_name='ecg', l2_dense = 0.01, l2_cnn = 0.01, glrt = glorot_uniform(seed=4), classes=2, is_unimodal=True):

    if is_unimodal:
        X_in = Input(X_in_shpe[0])
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

    else: X_in = X_in_shpe

    X = conv_blk1D(X_in, [32, 32], [64, 64], [1, 3], 'stage1', mod_name, l2_cnn)
    X = conv_blk1D(X, [64, 64], [32, 32], [1, 3], 'stage2', mod_name, l2_cnn)
    X = conv_blk1D(X, [128, 128], [17, 17], [1, 3], 'stage3', mod_name, l2_cnn)
    X = conv_blk1D(X, [256, 256], [7, 7], [1, 3], 'stage4', mod_name, l2_cnn)

    flatten_X = Flatten()(X)

    dense_X = Dense(512, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense)(flatten_X)
    
    dense_X= Dense(512, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense, name=mod_name + "bf_merge")(dense_X)

    if is_unimodal:
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(dense_X)
        model = Model(inputs = X_in, outputs = out)
        return model

    return dense_X

def unimodaleeg(X_in_shpe, mod_name='ecg', l2_dense = 0.01, l2_cnn = 0.01, glrt = glorot_uniform(seed=4), classes=2, is_unimodal=True):

    if is_unimodal:
        X_in = Input(X_in_shpe[0])
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

    X = conv_blk1D(X_in, [32, 32], [32, 32], [1, 3], 'stage1', mod_name, l2_cnn)
    X = conv_blk1D(X, [64, 64], [32, 32], [1, 3], 'stage2', mod_name, l2_cnn)
    X = conv_blk1D(X, [128, 128], [17, 17], [1, 3], 'stage3', mod_name, l2_cnn)
    X = conv_blk1D(X, [256, 256], [7, 7], [1, 3], 'stage4', mod_name, l2_cnn)

    flatten_X = Flatten()(X)

    dense_X = Dense(512, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense)(flatten_X)
    
    dense_X= Dense(512, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense, name=mod_name + "bf_merge")(dense_X)

    if is_unimodal:
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(dense_X)
        model = Model(inputs = X_in, outputs = out)
        return model

    return dense_X

def gze_arch(X_in_shpe, mod_name='gze',
            l2_dense = 0.01, l2_cnn = 0.01,
            glrt = glorot_uniform(seed=4), classes=2, is_unimodal=True):

    if is_unimodal:
        X_in = Input(X_in_shpe[0])
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

    else: X_in = X_in_shpe

    X = conv_blk1D(X_in, [32, 32], [64, 64], [1, 3], 'stage1', mod_name, l2_cnn)
    X = conv_blk1D(X, [64, 64], [32, 32], [1, 3], 'stage2', mod_name, l2_cnn)
    # X = conv_blk1D(X, [128, 128], [17, 17], [1, 3], 'stage3', mod_name, l2_cnn)
    # X = conv_blk1D(X, [256, 256], [7, 7], [1, 3], 'stage4', mod_name, l2_cnn)

    flatten_X = Flatten()(X)

    dense_X = Dense(128, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense)(flatten_X)
    
    dense_X= Dense(128, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense, name=mod_name + "bf_merge")(dense_X)

    if is_unimodal:
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(dense_X)
        model = Model(inputs = [X_in], outputs = out)
        return model

    return dense_X

def eeg_arch(X_in_shpe, mod_name='gze',
            l2_dense = 0.01, l2_cnn = 0.01,
            glrt = glorot_uniform(seed=4), classes=2, is_unimodal=True):

    if is_unimodal:
        X_in = Input(X_in_shpe[0])
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

    else: X_in = X_in_shpe

    X = conv_blk1D(X_in, [32, 32], [32, 32], [1, 3], 'stage1', mod_name, l2_cnn)
    X = conv_blk1D(X, [64, 64], [32, 32], [1, 3], 'stage2', mod_name, l2_cnn)
    X = conv_blk1D(X, [128, 128], [17, 17], [1, 3], 'stage3', mod_name, l2_cnn)
    X = conv_blk1D(X, [256, 256], [7, 7], [1, 3], 'stage4', mod_name, l2_cnn)

    flatten_X = Flatten()(X)

    dense_X = Dense(128, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense)(flatten_X)
    
    dense_X= Dense(64, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense, name=mod_name + "bf_merge")(dense_X)

    if is_unimodal:
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(dense_X)
        model = Model(inputs = [X_in], outputs = out)
        return model

    return dense_X

def bimodal_Kfold(input_shape=[(2560, 1), (2560, 3)], mod_name=['ecg', 'eda'], classes=2, is_unimodal=False):
    with tf.device('/device:GPU:0'):
                 
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

        is_unimodal = False

        X_in_1 = Input(input_shape[0])
        X_in_2 = Input(input_shape[1])
                    
        print(X_in_1.shape)
        print(X_in_2.shape)

        model_inputs = [X_in_1, X_in_2]

        if mod_name[0] == 'eeg':
            X_1 = eeg_arch(X_in_1, mod_name[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_1 = unimodal_Kfold(X_in_1, mod_name[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

        if mod_name[1] == 'eeg':
            X_2 = eeg_arch(X_in_2, mod_name[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_2 = unimodal_Kfold(X_in_2, mod_name[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)            

        merged = concatenate([X_1, X_2])

        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = model_inputs, outputs = out)
        return model


def trimodal_Kfold(input_shape=[(2560, 1), (2560, 3), (2560, 4)], mod_name=['ecg', 'eda', 'eeg'], classes=2, is_unimodal=False):
    with tf.device('/device:GPU:0'):
                 
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

        is_unimodal = False

        X_in_1 = Input(input_shape[0])
        X_in_2 = Input(input_shape[1])
        X_in_3 = Input(input_shape[2])
                    
        print(X_in_1.shape)
        print(X_in_2.shape)
        print(X_in_3.shape)

        model_inputs = [X_in_1, X_in_2, X_in_3]

        if mod_name[0] == 'eeg':
            X_1 = eeg_arch(X_in_1, mod_name[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_1 = unimodal_Kfold(X_in_1, mod_name[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

        if mod_name[1] == 'eeg':
            X_2 = eeg_arch(X_in_2, mod_name[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_2 = unimodal_Kfold(X_in_2, mod_name[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)            

        if mod_name[2] == 'eeg':
            X_3 = eeg_arch(X_in_3, mod_name[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_3 = unimodal_Kfold(X_in_3, mod_name[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

        merged = concatenate([X_1, X_2, X_3])

        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = model_inputs, outputs = out)
        
        return model

def quadmodal_Kfold(input_shape=[(2560, 1), (2560, 3), (2560, 4), (2560, 2)], mod_name=['ecg', 'eda', 'eeg', 'gze'], classes=2, is_unimodal=False):
    with tf.device('/device:GPU:0'):
                 
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

        is_unimodal = False

        X_in_1 = Input(input_shape[0])
        X_in_2 = Input(input_shape[1])
        X_in_3 = Input(input_shape[2])
        X_in_4 = Input(input_shape[3])
                    
        print(X_in_1.shape)
        print(X_in_2.shape)
        print(X_in_3.shape)
        print(X_in_4.shape)

        model_inputs = [X_in_1, X_in_2, X_in_3, X_in_4]

        if mod_name[0] == 'eeg':
            X_1 = eeg_arch(X_in_1, mod_name[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_1 = unimodal_Kfold(X_in_1, mod_name[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

        if mod_name[1] == 'eeg':
            X_2 = eeg_arch(X_in_2, mod_name[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_2 = unimodal_Kfold(X_in_2, mod_name[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)            

        if mod_name[2] == 'eeg':
            X_3 = eeg_arch(X_in_3, mod_name[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_3 = unimodal_Kfold(X_in_3, mod_name[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

        if mod_name[3] == 'eeg':
            X_4 = eeg_arch(X_in_4, mod_name[3], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
        else: 
            X_4 = unimodal_Kfold(X_in_4, mod_name[3], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

        merged = concatenate([X_1, X_2, X_3, X_4])

        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = model_inputs, outputs = out)
        
        return model


def multimodal_classifier(input_shape=[(2560, 1), (2560, 3)], classes=2, modality_names=['ecg', 'eda']):
    with tf.device('/device:GPU:0'):
                 
        glrt = glorot_uniform(seed=4)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

        is_unimodal = False

        if len(input_shape) == 2:
            X_in_1 = Input(input_shape[0])
            X_in_2 = Input(input_shape[1])
                        
            print(X_in_1.shape)
            print(X_in_2.shape)

            model_inputs = [X_in_1, X_in_2]

            if modality_names[0] == 'gze':
                X_1 = gze_arch(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[0] == 'eeg':
                X_1 = eeg_arch(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_1 = unimodal(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            if modality_names[1] == 'gze':
                X_2 = gze_arch(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[1] == 'eeg':
                X_2 = eeg_arch(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_2 = unimodal(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)    

            # X_1 = unimodal(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            # X_2 = unimodal(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            merged = concatenate([X_1, X_2])

        if len(input_shape) == 3:
            X_in_1 = Input(input_shape[0])
            X_in_2 = Input(input_shape[1])
            X_in_3 = Input(input_shape[2])

            print(X_in_1.shape)
            print(X_in_2.shape)
            print(X_in_3.shape)

            model_inputs = [X_in_1, X_in_2, X_in_3]

            if modality_names[0] == 'gze':
                X_1 = gze_arch(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[0] == 'eeg':
                X_1 = eeg_arch(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_1 = unimodal(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            if modality_names[1] == 'gze':
                X_2 = gze_arch(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[1] == 'eeg':
                X_2 = eeg_arch(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_2 = unimodal(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            if modality_names[2] == 'gze':
                X_3 = gze_arch(X_in_3, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[2] == 'eeg':
                X_3 = eeg_arch(X_in_3, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_3 = unimodal(X_in_3, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            # X_1 = unimodal(X_1_ip, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            # X_2 = unimodal(X_2_ip, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            # X_3 = unimodal(X_3_ip, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            merged = concatenate([X_1, X_2, X_3])

        if len(input_shape) == 4:
            X_in_1 = Input(input_shape[0])
            X_in_2 = Input(input_shape[1])
            X_in_3 = Input(input_shape[2])
            X_in_4 = Input(input_shape[3])

            print(X_in_1.shape)
            print(X_in_2.shape)
            print(X_in_3.shape)
            print(X_in_4.shape)

            model_inputs = [X_in_1, X_in_2, X_in_3, X_in_4]

            if modality_names[0] == 'gze':
                X_1 = gze_arch(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[0] == 'eeg':
                X_1 = eeg_arch(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_1 = unimodal(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            if modality_names[1] == 'gze':
                X_2 = gze_arch(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[1] == 'eeg':
                X_2 = eeg_arch(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_2 = unimodal(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            if modality_names[2] == 'gze':
                X_3 = gze_arch(X_in_3, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[2] == 'eeg':
                X_3 = eeg_arch(X_in_3, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_3 = unimodal(X_in_3, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            if modality_names[3] == 'gze':
                X_4 = gze_arch(X_in_4, modality_names[3], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            elif modality_names[3] == 'eeg':
                X_4 = eeg_arch(X_in_4, modality_names[3], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            else: 
                X_4 = unimodal(X_in_4, modality_names[3], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            # X_1 = unimodal(X_1_ip, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            # X_2 = unimodal(X_2_ip, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            # X_3 = unimodal(X_3_ip, modality_names[2], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            # X_4 = unimodal(X_4_ip, modality_names[3], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            merged = concatenate([X_1, X_2, X_3, X_4])

        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = model_inputs, outputs = out)
        
        return model