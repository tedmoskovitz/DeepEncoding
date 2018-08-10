"""
Ted Moskovitz, 2018
Keras Models for Neural Encoding
"""
import numpy as np
import scipy.io as spio 
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, GRU, CuDNNLSTM, Flatten, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Reshape, Lambda
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from tf_utils import neglogli, poiss_full, get_bps, r2, restore_performance_checkpt, color, L2_func
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import keras

# for functional model
from keras.layers import Input, Embedding, Dense
from keras.models import Model

def gen_fc_model(H, dim=20, L=0, verbose=True, LNP=0, filts=None, trainable=False): 
    """
    Generate a fully-connected network (MLP). 
    
    Args: 
        H: hidden layer size 
        dim: stimulus (input) dimension
        L: depth
        verbose: whether to print a summary of the model upon construction
        LNP: number of initial filters for DNN-LNP model; using 0 results in a standard DNN
        filts: initialization of filters
        trainable: whether initial DNN-LNP filters are trainable
        
    Returns: 
        the compiled Keras model
    """
    model = Sequential()
    if LNP > 0: # pre-DNN filters
        if filts is None: model.add(Dense(LNP, input_dim=dim));
        else: model.add(Dense(LNP, weights=[filts, np.zeros(LNP)],
                              trainable=trainable, input_dim=dim, name='istac_filts'));
        model.add(Dense(H, activation='softplus'))
    else:
        model.add(Dense(H, input_dim=dim))
        
    model.add(Activation('softplus'))
    if H >= 50 and H < 100: 
        model.add(Dropout(.1))
    elif H >= 100:
         model.add(Dropout(.2))

    for _ in range(L):
        model.add(Dense(H))
        model.add(Activation('softplus'))
        if H >= 50 and H < 100: 
            model.add(Dropout(.2))
        elif H >= 100:
            model.add(Dropout(.4))

    model.add(Dense(1, activation='softplus'))
    
    adam = Adam(lr=.00005, decay=1e-6)
    model.compile(loss=poiss_full, optimizer=adam) #metrics=['mean_squared_error']

    if verbose:
        model.summary()
        
    return model

def gen_rnn_model(H, dim=20, n_frames=16, L=0, verbose=True, use_cudnn=True, LNP=0):
    """
    Generate an LSTM network.
    
    Args: 
        H: hidden state size 
        dim: stimulus (input) dimension
        n_frames: frames of history to consider
        L: depth of network
        verbose: whether to print a model summary
        use_cudnn: whether to use CuDNNLSTM layer, optimized for training on a GPU
        LNP: input filters; using 0 results in a standard DNN
        
    Returns: 
        compiled Keras model
    """
    model = Sequential()
    if L == 0: ret_seqs = False;
    else: ret_seqs = True;
        
    if use_cudnn: lstm = CuDNNLSTM;
    else: lstm = LSTM;
        
    if LNP > 0: 
        model.add(Dense(LNP, input_dim=dim)); #??
        model.add(lstm(H, input_shape=(n_frames, LNP), return_sequences=ret_seqs))
    else: 
        model.add(lstm(H, input_shape=(n_frames, dim), return_sequences=ret_seqs))
    
    for i in range(L):
        if i == L-1: ret_seqs = False;
        else: ret_seqs = True
        model.add(lstm(H, return_sequences=ret_seqs)) 
    
    model.add(Dense(1, activation='softplus'))
    
    if verbose:
        model.summary()
    
    adam = Adam(lr=.001, decay=1e-6)

    model.compile(loss=poiss_full, optimizer=adam) 
    return model

def gen_cnn_model(H, L=0, dim=1, n_frames=16, ksize=7, verbose=True, LNP=0, filts=None, trainable=False):
    """
    Generate a 1-D CNN.
    
    Args: 
        H: number of kernels per layer
        L: depth of network
        dim: stimulus (input) dimension
        n_frames: length of history (convolving over time, so equivalent to # of frames)
        ksize: length of 1-D kernel
        verbose: whether to print a model summary
        LNP: number of input filters; using 0 results in a standard DNN
        filts: initial values for filters 
        
    Returns: 
        compiled Keras model
    """
    model = Sequential()
    
    if LNP > 0:
        model.add(Conv1D(LNP, ksize, input_shape=(n_frames,dim)))
        #model.add(Reshape((n_frames, LNP))
        model.add(Conv1D(H, ksize, activation='softplus', padding='same')) #, input_shape=(n_frames, LNP)
    else:
        model.add(Conv1D(H, ksize, activation='softplus', padding='same', input_shape=(n_frames, dim)))
    model.add(Conv1D(H, ksize, activation='softplus', padding='same'))
    
    if L > 0:
        model.add(MaxPooling1D(2))
        if H > 16: model.add(Dropout(.3));
    
    H_curr = H
    for i in range(L):
        model.add(Conv1D(H_curr, ksize-2*(i+1), padding='same', activation='softplus')) # 2x kernels after each downsample
        model.add(Conv1D(H_curr, ksize-2*(i+1), padding='same', activation='softplus'))
        model.add(MaxPooling1D(2))
        drop = 0.0
        if H > 16: drop = .3;
        if H > 32: drop = .5;
        model.add(Dropout(drop))
        H_curr *= 2
        
    model.add(GlobalAveragePooling1D())
    if H > 16 and H < 64 and L > 0: model.add(Dropout(0.3)); 
    if H > 32 and L > 0: model.add(Dropout(0.5)); 
    model.add(Dense(1, activation='softplus'))
    
    if verbose:
        model.summary()
    
    if L >= 2: eta = .0001
    else: eta = .001
    adam = Adam(lr=eta, decay=1e-6)

    model.compile(loss=poiss_full, optimizer=adam) 
    return model


def gen_ES_model(e_filts, s_filts, H, dim=20, L=0, verbose=True, trainable=False, lr=0.0001):
    """
    Generate Excitatory-Suppressive (ES) Model
    
    Args: 
        e_filts: excitatory filters
        s_filts: suppressive filters
        H: dimension of hidden layer for MLP nonlinearity
        dim: stimulus (input) dimension
        L: depth of MLP nonlinearity
        verbose: whether to print a model summary
        trainable: whether input filters are trainable
        
    Returns: 
        compiled Keras model
    """
    nE = e_filts.shape[1] # number of excitatory filters 
    nS = s_filts.shape[1] # number of suppressive filters
    
    # process excitatory and suppressive streams
    inpt = Input(shape=(dim,), dtype='float32', name='input_stim')
    
    E = Dense(nE, weights=[e_filts, np.zeros(nE)], trainable=trainable, name='exc_filter')(inpt) 
    S = Dense(nS, weights=[s_filts, np.zeros(nS)], trainable=trainable, name='sup_filter')(inpt) 
    
    E = Dense(nE, name='excitatory_wts')(E)
    S = Dense(nS, name='suppressive_wts')(S)
    
    E = Lambda(L2_func)(E) 
    S = Lambda(L2_func)(S)
    
    # feed two streams as 2D input to network 
    x = keras.layers.concatenate([E, S])
    for _ in range(L):
        x = Dense(H, activation='softplus')(x)
        if H >= 100: x = Dropout(0.4)(x);
        if H > 50 and H < 100: x = Dropout(0.2)(x); 
            
    out = Dense(1, activation='softplus')(x)
    
    model = Model(inputs=[inpt], outputs=[out])
    adam = Adam(lr=lr, decay=1e-6)
    model.compile(loss=poiss_full, optimizer=adam)
    
    return model
    
    
