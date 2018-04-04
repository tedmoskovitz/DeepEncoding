import numpy as np
import scipy.io as spio 
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, GRU, CuDNNLSTM, Flatten, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import RMSprop, SGD, Adam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping

def gen_fc_model(H, dim=20, L=0, verbose=True): 
    """generate a full connected network with hidden size H and depth L"""
    model = Sequential()
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
    
    adam = Adam(lr=.0001, decay=1e-6)
    model.compile(loss=poiss_full, optimizer=adam) #metrics=['mean_squared_error']

    if verbose:
        model.summary()
        
    return model

def gen_rnn_model(H, dim=20, n_frames=16, L=0, verbose=True, use_cudnn=True):
    """generate an LSTM network"""
    model = Sequential()
    if L == 0: ret_seqs = False;
    else: ret_seqs = True;
    
        
    if use_cudnn: lstm = CuDNNLSTM;
    else: lstm = LSTM;
        
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

def gen_cnn_model(H, L=0, dim=1, n_frames=16, ksize=7, verbose=True):
    """generate a 1D CNN"""
    model = Sequential()
    
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
    
    if L > 2: eta = .0001
    else: eta = .001
    adam = Adam(lr=eta, decay=1e-6)

    model.compile(loss=poiss_full, optimizer=adam) 
    return model
