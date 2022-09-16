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

class Conv_blc(tf.keras.layers.Layer):
    
    def __init__(self, filters, f, strides, stage, stype, l2_cnn):
        super(Conv_blc, self).__init__()
        self.F1 = filters[0]
        self.F2 = filters[1]
        
        self.f1 = f[0]
        self.f2 = f[1]
        
        self.s1 = strides[0]
        self.s2 = strides[1]
        
        self.conv_name = 'conv_' + str(stage) + "_" + str(stype)
        self.batch_name = 'bn_' + str(stage) + "_" + str(stype)
        self.act_name = 'act_' + str(stage) + "_" + str(stype)
        self.maxpool_name = 'mp_' + str(stage) + "_" + str(stype)
        
        self.conv1 = Conv1D(self.F1, self.f1, strides=self.s1, name= self.conv_name + 'a',
                            kernel_regularizer = l2_cnn, padding = 'same',
                            kernel_initializer = glorot_uniform(seed=0))
        self.conv2 = Conv1D(self.F2, self.f2, strides=self.s2, name= self.conv_name + 'b',
                            kernel_regularizer=l2_cnn, padding = 'same',
                            kernel_initializer=glorot_uniform(seed=0))
        
        self.bn = BatchNormalization(name=self.batch_name)
        self.mp = MaxPooling1D(2, strides=2, name=self.maxpool_name)
        
    def call(self, X_ip, training=False):
        x = self.conv1(X_ip)
        x = self.conv2(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        
        return x

class Attention_layer(tf.keras.layers.Layer):
    
    def __init__(self, hidden_size):
        super(Attention_layer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.u_w = tf.Variable(initial_value=w_init(shape=(hidden_size, 1)))
        
    def call(self, inputs):
        x_dot = tf.tensordot(inputs, self.u_w, axes=1)
        alpha = tf.nn.softmax(x_dot, axis = 2)
        alpha_0 = alpha[:, :, 0, :]
        alpha_1 = alpha[:, :, 1, :]
        return alpha_0, alpha_1
    
# creating a class of attention
class Attention2():
    def __init__(self):
        self.v = None
    
    def self_attention(self, layers, name = ''):
        """
        :param inputs_a: audio input (B, T, dim)
        :param inputs_v: video input (B, T, dim)
        :param name: scope name
        :return:
        """

        if self.v is None: # layer[0] shape is None, 639, 64
            with tf.device('/device:GPU:0'):
                batch_dim = layers[0].get_shape()[0]
                timestep_dim = layers[0].get_shape()[1]
                hidden_dim = layers[0].get_shape()[2]

                inputs_a = tf.expand_dims(layers[0], axis=2) # B, T, 1, Dims
                inputs_v = tf.expand_dims(layers[1], axis=2) # B, T, 1, Dims

                inputs = tf.concat([inputs_a, inputs_v], axis=2) # inputs = (B, T, C, Dims)
                share_param = True
                
                kernel_init = glorot_uniform(seed=2222)
                if share_param:
                    scope_name = 'self_attn'

                inputs_transpose = tf.transpose(inputs, [0, 1, 3, 2]) # B, T, Dims, C == 2
                hidden_dense = inputs_transpose.get_shape()[-1]
                dense = Dense(hidden_dense, kernel_initializer=kernel_init)
                x_proj = dense(inputs_transpose)
                x_proj = tf.nn.relu(x_proj)
                x_proj_tran = tf.transpose(x_proj, (0, 1, 3, 2))
                hidden_size = x_proj_tran.shape[-1] # dims
                
                attn_layer = Attention_layer(hidden_size)
                alpha_0, alpha_1 = attn_layer(x_proj_tran)
                
                output_a = tf.math.multiply(layers[0], alpha_0)
                output_v = tf.math.multiply(layers[1], alpha_1)
                out_a = tf.concat([layers[0], output_v], axis=-1, name='conc_ecg')
                out_v = tf.concat([layers[1], output_a], axis=-1, name='conc_eda')

                return out_a, out_v

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

def mega_model(input_shape=[(2560, 1), (2560, 3)], 
                attx_type='I', attx_st='one', classes = 2):
    with tf.device('/device:GPU:0'):
        eda_l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)
        eda_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.00)
        ecg_l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)
        ecg_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.00)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)

        seed2 = 4
        glrt = glorot_uniform(seed=4)
        # Define the input as a tensor with shape input_shape
        
        X_ecg_ip = Input(input_shape[0])
        X_eda_ip = Input(input_shape[1])
        
        print(X_ecg_ip.shape)
        print(X_eda_ip.shape)
        
        ''' Stage 1 '''

        X_ecg = conv_blk1D(X_ecg_ip, [32, 32], [64, 64], [1, 3], 'stage1', 'ecg', ecg_l2_cnn)
        X_eda = conv_blk1D(X_eda_ip, [32, 32], [64, 64], [1, 3], 'stage1', 'eda', eda_l2_cnn)

        attx = Attention2()
        if attx_st in ['one', 'one_two', 'one_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = conv_blk1D(X_ecg, [64, 64], [32, 32], [1, 3], 'stage2', 'ecg', ecg_l2_cnn)
        X_eda = conv_blk1D(X_eda, [64, 64], [32, 32], [1, 3], 'stage2', 'eda', eda_l2_cnn)
        
        if attx_st in ['two', 'one_two', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = conv_blk1D(X_ecg, [128, 128], [17, 17], [1, 3], 'stage3', 'ecg', ecg_l2_cnn)
        X_eda = conv_blk1D(X_eda, [128, 128], [17, 17], [1, 3], 'stage3', 'eda', eda_l2_cnn)
        
        if attx_st in ['three', 'one_three', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = conv_blk1D(X_ecg, [256, 256], [7, 7], [1, 3], 'stage4', 'ecg', ecg_l2_cnn)
        X_eda = conv_blk1D(X_eda, [256, 256], [7, 7], [1, 3], 'stage4', 'eda', eda_l2_cnn)

        flatten_ecg = Flatten()(X_ecg)
        flatten_eda = Flatten()(X_eda)

        dense_ecg = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense)(flatten_ecg)
        
        dense_ecg = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense, name="ecg_bf_merge")(dense_ecg)

        dense_eda = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense)(flatten_eda)
        
        dense_eda = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense, name="eda_bf_merge")(dense_eda)
        
        merged = concatenate([dense_ecg, dense_eda])
        
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = [X_ecg_ip, X_eda_ip], outputs = out)
        
        return model


def mega_model_ecg(input_shape=[(2560, 1)], 
                attx_type='I', attx_st='one', classes = 2):
    with tf.device('/device:GPU:0'):
        eda_l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)
        eda_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.00)
        ecg_l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)
        ecg_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.00)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)

        seed2 = 4
        glrt = glorot_uniform(seed=4)
        # Define the input as a tensor with shape input_shape
        
        X_ecg_ip = Input(input_shape[0])
        # X_eda_ip = Input(input_shape[1])
        
        print(X_ecg_ip.shape)
        # print(X_eda_ip.shape)
        
        ''' Stage 1 '''

        X_ecg = conv_blk1D(X_ecg_ip, [32, 32], [64, 64], [1, 3], 'stage1', 'ecg', ecg_l2_cnn)
        X_ecg = conv_blk1D(X_ecg, [64, 64], [32, 32], [1, 3], 'stage2', 'ecg', ecg_l2_cnn)
        X_ecg = conv_blk1D(X_ecg, [128, 128], [17, 17], [1, 3], 'stage3', 'ecg', ecg_l2_cnn)
        X_ecg = conv_blk1D(X_ecg, [256, 256], [7, 7], [1, 3], 'stage4', 'ecg', ecg_l2_cnn)

        flatten_ecg = Flatten()(X_ecg)
        dense_ecg = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense)(flatten_ecg)
        
        dense_ecg = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense, name="ecg_bf_merge")(dense_ecg)
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = l2_dense)(dense_ecg)
        model = Model(inputs = [X_ecg_ip], outputs = out)
        return model        

def mega_model_test(input_shape=[(2560, 1), (2560, 3)], 
                attx_type='I', attx_st='one', classes = 2):
    with tf.device('/device:GPU:0'):
        eda_l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)
        eda_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.00)
        ecg_l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)
        ecg_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.00)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.00)

        seed2 = 4
        glrt = glorot_uniform(seed=4)
        # Define the input as a tensor with shape input_shape
        
        X_ecg_ip = Input(input_shape[0])
        X_eda_ip = Input(input_shape[1])
        
        print(X_ecg_ip.shape)
        print(X_eda_ip.shape)
        
        ''' Stage 1 '''

        X_ecg = conv_blk1D(X_ecg_ip, [32, 32], [64, 64], [1, 3], 'stage1', 'ecg', ecg_l2_cnn)
        X_eda = conv_blk1D(X_eda_ip, [32, 32], [64, 64], [1, 3], 'stage1', 'eda', eda_l2_cnn)

        # attx = Attention2()
        # if attx_st in ['one', 'one_two', 'one_three', 'all']:
        #     if attx_type == 'III':
        #         X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
        #     elif attx_type == 'I':
        #         _, X_eda = attx.self_attention([X_ecg, X_eda])
        #     elif attx_type == 'II':
        #         X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        # X_ecg = conv_blk1D(X_ecg, [64, 64], [32, 32], [1, 3], 'stage2', 'ecg', ecg_l2_cnn)
        # X_eda = conv_blk1D(X_eda, [64, 64], [32, 32], [1, 3], 'stage2', 'eda', eda_l2_cnn)
        
        # if attx_st in ['two', 'one_two', 'two_three', 'all']:
        #     if attx_type == 'III':
        #         X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
        #     elif attx_type == 'I':
        #         _, X_eda = attx.self_attention([X_ecg, X_eda])
        #     elif attx_type == 'II':
        #         X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        # X_ecg = conv_blk1D(X_ecg, [128, 128], [17, 17], [1, 3], 'stage3', 'ecg', ecg_l2_cnn)
        # X_eda = conv_blk1D(X_eda, [128, 128], [17, 17], [1, 3], 'stage3', 'eda', eda_l2_cnn)
        
        # if attx_st in ['three', 'one_three', 'two_three', 'all']:
        #     if attx_type == 'III':
        #         X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
        #     elif attx_type == 'I':
        #         _, X_eda = attx.self_attention([X_ecg, X_eda])
        #     elif attx_type == 'II':
        #         X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        # X_ecg = conv_blk1D(X_ecg, [256, 256], [7, 7], [1, 3], 'stage4', 'ecg', ecg_l2_cnn)
        # X_eda = conv_blk1D(X_eda, [256, 256], [7, 7], [1, 3], 'stage4', 'eda', eda_l2_cnn)

        flatten_ecg = Flatten()(X_ecg)
        flatten_eda = Flatten()(X_eda)

        dense_ecg = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense)(flatten_ecg)
        
        dense_ecg = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense, name="ecg_bf_merge")(dense_ecg)

        dense_eda = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense)(flatten_eda)
        
        dense_eda = Dense(512, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense, name="eda_bf_merge")(dense_eda)
        
        merged = concatenate([dense_ecg, dense_eda])
        
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = [X_ecg_ip, X_eda_ip], outputs = out)
        
        return model


def conv_blk1D_(X_eda_ip, filters, f, strides, stage, stype, l2_cnn):
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
    # X = conv_blk1D(X, [64, 64], [32, 32], [1, 3], 'stage2', mod_name, l2_cnn)
    # X = conv_blk1D(X, [128, 128], [17, 17], [1, 3], 'stage3', mod_name, l2_cnn)
    # X = conv_blk1D(X, [256, 256], [7, 7], [1, 3], 'stage4', mod_name, l2_cnn)

    flatten_X = Flatten()(X)

    dense_X = Dense(512, activation = 'relu', 
                kernel_initializer = glrt,
                kernel_regularizer = l2_dense)(flatten_X)
    
    # dense_X= Dense(512, activation = 'relu', 
    #             kernel_initializer = glrt,
    #             kernel_regularizer = l2_dense, name=mod_name + "bf_merge")(dense_X)

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

def multimodal_classifier1(input_shape=[(2560, 1), (2560, 3)], classes=2, modality_names=['ecg', 'eda']):
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

            X_1 = unimodal(X_in_1, modality_names[0], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            X_2 = unimodaleeg(X_in_2, modality_names[1], l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            merged = concatenate([X_1, X_2])

        if len(input_shape) == 3:
            X_ecg_ip = Input(input_shape[0])
            X_eda_ip = Input(input_shape[1])
            X_eeg_ip = Input(input_shape[2])

            print(X_ecg_ip.shape)
            print(X_eda_ip.shape)
            print(X_eeg_ip.shape)

            model_inputs = [X_ecg_ip, X_eda_ip, X_eeg_ip]

            X_ecg = unimodal(X_ecg_ip, 'ecg', l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            X_eda = unimodal(X_eda_ip, 'eda', l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)
            X_eeg = unimodal(X_eeg_ip, 'eeg', l2_dense, l2_cnn, glrt, classes=classes, is_unimodal=is_unimodal)

            merged = concatenate([X_ecg, X_eda, X_eeg])

        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = model_inputs, outputs = out)
        
        return model