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

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    kernel_init = glorot_uniform(seed=0)

    # First component of main path
    X = Conv1D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + 'a',
               kernel_initializer = kernel_init)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + 'a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + 'b',
               kernel_initializer = kernel_init)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + 'b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + 'c',
               kernel_initializer = kernel_init)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + 'c')(X)
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, filters, f, s, stage, l2_cnn, is_identity=False):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res_' + str(stage)
    bn_name_base = 'bn_' + str(stage) 
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value
    X_shortcut = X

    kernel_init = glorot_uniform(seed=0)

    ##### MAIN PATH #####

    padding_type = 'valid'
    if is_identity:
        s = 1
        conv_name_base = 'iden_' + conv_name_base
        bn_name_base = 'iden_' + bn_name_base

    else:
        conv_name_base = 'conv_' + conv_name_base
        bn_name_base = 'conv_' + bn_name_base

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv1D(filters = F2, kernel_size = 1, strides = s, padding = 'valid', name = conv_name_base + '_short',
                            kernel_initializer = kernel_init, kernel_regularizer = l2_cnn)(X_shortcut)
        X_shortcut = BatchNormalization(axis = -1, name = bn_name_base + '_short')(X_shortcut)

    # First component of main path 
    X = Conv1D(filters = F1, kernel_size = 1, strides = s, padding = padding_type, name = conv_name_base + '_a',
               kernel_initializer = kernel_init, kernel_regularizer = l2_cnn)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + '_a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv1D(filters = F1, kernel_size = f, strides = 1, padding = 'same', name = conv_name_base + '_b',
               kernel_initializer = kernel_init, kernel_regularizer = l2_cnn)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + '_b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv1D(filters = F2, kernel_size = 1, strides = 1, padding = 'valid', name = conv_name_base + '_c',
               kernel_initializer = kernel_init, kernel_regularizer = l2_cnn)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + '_c')(X)
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def res_block(X_input, filters, f, strides, stage, stype, l2_cnn):

    conv_name = str(stage) + "_" + str(stype)

    X = convolutional_block(X_input, filters, f, strides, conv_name, l2_cnn, is_identity = False)
    X = convolutional_block(X, filters, f, strides, conv_name, l2_cnn, is_identity = True)

    return X

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

def mega_resnet(input_shape=[(2560, 1), (2560, 3)], 
                attx_type='I', attx_st='one', classes = 2):
    with tf.device('/device:GPU:0'):
        eda_l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        eda_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)
        ecg_l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        ecg_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)

        seed2 = 4
        glrt = glorot_uniform(seed=4)
        # Define the input as a tensor with shape input_shape
        X_ecg_ip = Input(input_shape[0])
        X_eda_ip = Input(input_shape[1])
        
        print(X_ecg_ip.shape)
        print(X_eda_ip.shape)
        
        ''' Stage 1 '''

        X_ecg = res_block(X_ecg_ip, [32, 64], 64, 7, 'stage1', 'ecg', ecg_l2_cnn)
        X_eda = res_block(X_eda_ip, [32, 64], 64, 7, 'stage1', 'eda', eda_l2_cnn)

        print(X_ecg.shape)
        print(X_eda.shape)

        attx = Attention2()
        if attx_st in ['one', 'one_two', 'one_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [64, 128], 32, 3, 'stage2', 'ecg', ecg_l2_cnn) # kernel size  was 32 here!
        X_eda = res_block(X_eda, [64, 128], 32, 3, 'stage2', 'eda', eda_l2_cnn)

        if attx_st in ['two', 'one_two', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [128, 256], 17, 3, 'stage3', 'ecg', ecg_l2_cnn) # kernel size  was 17 here!
        X_eda = res_block(X_eda, [128, 256], 17, 3, 'stage3', 'eda', eda_l2_cnn)

        if attx_st in ['three', 'one_three', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [256, 512], 7, 3, 'stage4', 'ecg', ecg_l2_cnn) # kernel size  was 7 here!
        X_eda = res_block(X_eda, [256, 512], 7, 3, 'stage4', 'eda', eda_l2_cnn)

        flatten_ecg = Flatten()(X_ecg)
        flatten_eda = Flatten()(X_eda)
        
        dense_ecg = Dense(1024, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense)(flatten_ecg)
        
        dense_ecg = Dense(256, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense, name="ecg_bf_merge")(dense_ecg)

        dense_eda = Dense(1024, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense)(flatten_eda)
        
        dense_eda = Dense(256, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense, name="eda_bf_merge")(dense_eda)
        
        merged = concatenate([dense_ecg, dense_eda])
        
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = [X_ecg_ip, X_eda_ip], outputs = out)
       
        return model

def first_conv(X_ip, num_kernel, kernel_size, s, padin):
    X = Conv1D(num_kernel, kernel_size, s, padin)(X_ip)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)
    return 
    
def mega_resnet_ff(input_shape=[(2560, 1), (2560, 3)], 
                attx_type='I', attx_st='one', classes = 2):
    with tf.device('/device:GPU:0'):
        eda_l2_dense = tensorflow.keras.regularizers.l2(l = 0.0015)
        eda_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.0015)
        ecg_l2_dense = tensorflow.keras.regularizers.l2(l = 0.0015)
        ecg_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.0015)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.0015)

        seed2 = 4
        glrt = glorot_uniform(seed=4)
        # Define the input as a tensor with shape input_shape
        X_ecg_ip = Input(input_shape[0])
        X_eda_ip = Input(input_shape[1])
        
        print(X_ecg_ip.shape)
        print(X_eda_ip.shape)
        
        ''' Stage 1 '''
        X_ecg = res_block(X_ecg_ip, [32, 64], 64, 7, 'stage1', 'ecg', ecg_l2_cnn)
        X_eda = res_block(X_eda_ip, [32, 64], 64, 7, 'stage1', 'eda', eda_l2_cnn)

        attx = Attention2()
        if attx_st in ['one', 'one_two', 'one_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [64, 128], 32, 3, 'stage2', 'ecg', ecg_l2_cnn) # kernel size  was 32 here!
        X_eda = res_block(X_eda, [64, 128], 32, 3, 'stage2', 'eda', eda_l2_cnn)

        if attx_st in ['two', 'one_two', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [128, 256], 17, 3, 'stage3', 'ecg', ecg_l2_cnn) # kernel size  was 17 here!
        X_eda = res_block(X_eda, [128, 256], 17, 3, 'stage3', 'eda', eda_l2_cnn)

        if attx_st in ['three', 'one_three', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [256, 512], 7, 3, 'stage4', 'ecg', ecg_l2_cnn) # kernel size  was 7 here!
        X_eda = res_block(X_eda, [256, 512], 7, 3, 'stage4', 'eda', eda_l2_cnn)

        flatten_ecg = Flatten()(X_ecg)
        flatten_eda = Flatten()(X_eda)
        
        dense_ecg = Dense(128, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense)(flatten_ecg)
        
        dense_ecg = Dense(64, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense, name="ecg_bf_merge")(dense_ecg)

        dense_eda = Dense(128, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense)(flatten_eda)
        
        dense_eda = Dense(64, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense, name="eda_bf_merge")(dense_eda)
        
        merged = concatenate([dense_ecg, dense_eda])
        
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = [X_ecg_ip, X_eda_ip], outputs = out)
        return model


def ablation_model(input_shape=[(2560, 1)], classes = 2):
    with tf.device('/device:GPU:0'):
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)
        seed2 = 4
        glrt = glorot_uniform(seed=4)
        # Define the input as a tensor with shape input_shape
        X_ip = Input(input_shape[0])
        X = res_block(X_ip, [32, 64], 64, 7, 'stage1', '_X', l2_cnn)
        X = res_block(X, [64, 128], 32, 3, 'stage2', '_X', l2_cnn) # kernel size  was 32 here!
        X = res_block(X, [128, 256], 17, 3, 'stage3', '_X', l2_cnn) # kernel size  was 17 here!
        X = res_block(X, [256, 512], 7, 3, 'stage4', '_X', l2_cnn) # kernel size  was 7 here!
        flattenX = Flatten()(X)
        denseX = Dense(128, activation = 'relu', 
                    kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(flattenX)
        denseX = Dense(64, activation = 'relu', 
                    kernel_initializer = glrt,
                    kernel_regularizer = l2_dense, name="X_bf_merge")(denseX)
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(denseX)
        model = Model(inputs = [X_ip], outputs = out)
        return model

def ablation_modelECG(input_shape=[(2560, 1)], classes = 2):
    with tf.device('/device:GPU:0'):
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)

        seed2 = 4
        glrt = glorot_uniform(seed=4)
        mx = 4
        mk = 1
        dk = 1

        # Define the input as a tensor with shape input_shape
        X_ip = Input(input_shape[0])

        X = res_block(X_ip, [32*mx, 64*mx], 64*mk, 7, 'stage1', '_X', l2_cnn)
        # X = res_block(X, [64*mx, 128*mx], 32*mk, 3, 'stage2', '_X', l2_cnn) # kernel size  was 32 here!
        # X = res_block(X, [128*mx, 256*mx], 17*mk, 3, 'stage3', '_X', l2_cnn) # kernel size  was 17 here!
        # X = res_block(X, [256*mx, 512*mx], 7*mk, 3, 'stage4', '_X', l2_cnn) # kernel size  was 7 here!
        flattenX = Flatten()(X)
        denseX = Dense(64*dk, activation = 'relu', 
                    kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(flattenX)
        # denseX = Dense(64*dk, activation = 'relu', 
        #             kernel_initializer = glrt,
        #             kernel_regularizer = l2_dense, name="X_bf_merge")(denseX)
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glrt,
                    kernel_regularizer = l2_dense)(denseX)
        model = Model(inputs = [X_ip], outputs = out)
        
        return model

def first_conv(X_ip, num_kernel, kernel_size, s, padin):
    X = Conv1D(num_kernel, kernel_size, s, padin)(X_ip)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)
    return X

def mega_resnet_three(input_shape=[(2560, 1), (2560, 3)], 
                attx_type='I', attx_st='one', classes = 2):
    with tf.device('/device:GPU:0'):
        eda_l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        eda_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)
        ecg_l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)
        ecg_l2_cnn = tensorflow.keras.regularizers.l2(l = 0.001)
        l2_dense = tensorflow.keras.regularizers.l2(l = 0.001)

        seed2 = 4
        glrt = glorot_uniform(seed=4)
        # Define the input as a tensor with shape input_shape
        X_ecg_ip = Input(input_shape[0])
        X_eda_ip = Input(input_shape[1])
        
        print(X_ecg_ip.shape)
        print(X_eda_ip.shape)
        
        ''' Stage 1 '''

        X_ecg = first_conv(X_ecg_ip, 64, 7, 3, padding='same', kernel_initializer=glrt)
        X_eda = first_conv(X_eda_ip, 64, 7, 3, padding='same', kernel_initializer=glrt)

        X_ecg = res_block(X_ecg, [32, 64], 64, 7, 'stage1', 'ecg', ecg_l2_cnn)
        X_eda = res_block(X_eda, [32, 64], 64, 7, 'stage1', 'eda', eda_l2_cnn)

        print(X_ecg.shape)
        print(X_eda.shape)

        attx = Attention2()
        if attx_st in ['one', 'one_two', 'one_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [64, 128], 32, 3, 'stage2', 'ecg', ecg_l2_cnn) # kernel size  was 32 here!
        X_eda = res_block(X_eda, [64, 128], 32, 3, 'stage2', 'eda', eda_l2_cnn)

        if attx_st in ['two', 'one_two', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [128, 256], 17, 3, 'stage3', 'ecg', ecg_l2_cnn) # kernel size  was 17 here!
        X_eda = res_block(X_eda, [128, 256], 17, 3, 'stage3', 'eda', eda_l2_cnn)

        if attx_st in ['three', 'one_three', 'two_three', 'all']:
            if attx_type == 'III':
                X_ecg, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'I':
                _, X_eda = attx.self_attention([X_ecg, X_eda])
            elif attx_type == 'II':
                X_ecg, _ = attx.self_attention([X_ecg, X_eda])

        X_ecg = res_block(X_ecg, [256, 512], 7, 3, 'stage4', 'ecg', ecg_l2_cnn) # kernel size  was 7 here!
        X_eda = res_block(X_eda, [256, 512], 7, 3, 'stage4', 'eda', eda_l2_cnn)

        flatten_ecg = Flatten()(X_ecg)
        flatten_eda = Flatten()(X_eda)
        
        dense_ecg = Dense(256, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense)(flatten_ecg)
        
        dense_ecg = Dense(64, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = ecg_l2_dense, name="ecg_bf_merge")(dense_ecg)

        dense_eda = Dense(256, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense)(flatten_eda)
        
        dense_eda = Dense(64, activation = 'relu', 
                    kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = eda_l2_dense, name="eda_bf_merge")(dense_eda)
        
        merged = concatenate([dense_ecg, dense_eda])
        
        out = Dense(classes, activation = 'softmax', 
                    name = 'output', kernel_initializer = glorot_uniform(seed = seed2),
                    kernel_regularizer = l2_dense)(merged)
        model = Model(inputs = [X_ecg_ip, X_eda_ip], outputs = out)
        
        return model