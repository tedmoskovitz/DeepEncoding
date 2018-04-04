import numpy as np
import scipy.io as spio 
import h5py
from v1_data import V1
import os
import os.path
import tensorflow as tf
import scipy as sp
import scipy.misc as spm

def neglogli(preds, actual):
    """poisson negative log likelihood"""
    preds = np.reshape(preds, (-1,))
    nll = np.sum((preds - actual*np.log(preds + 1e-07) + np.log(sp.special.factorial(actual))))
    return nll

def poiss_full(y, yhat):
    """full poisson loss"""
    cost = tf.reduce_mean(tf.nn.log_poisson_loss(y, tf.log(yhat+1e-7), compute_full_loss=True))
    costp = tf.reduce_mean(tf.nn.log_poisson_loss(y, tf.log(y+1e-7), compute_full_loss=True))
    return cost 

def get_bps(model, data):
    """given a keras model, get single spike information"""
    preds = model.predict(data.X_test)

    avg_rate = float(np.sum(data.y_train)) / len(data.y_train)
    avg_pred = np.zeros_like(data.y_test)
    avg_pred.fill(avg_rate)
    nll_avg = neglogli(avg_pred, data.y_test.T)

    nll_model = neglogli(preds, data.y_test.T)
    
    nsp = np.sum(data.y_test)

    nll_perfect = neglogli(data.y_test.T, data.y_test.T)
    bps_perfect = abs((nll_perfect-nll_avg) / nsp / np.log(2))

    bps = abs((nll_model-nll_avg) / nsp / np.log(2))
    if nll_model > nll_avg:
        bps *= -1
    return bps, bps_perfect

def r2(preds, actual):
    """compute r^2 between predicted an actual spike trains"""
    preds = np.reshape(preds, (-1,))
    r = np.corrcoef(preds, actual)[0,1]
    rsq = r ** 2
    return r,rsq

def restore_performance_checkpt(path):
    """load a performance checkpoint, if it exists"""
    if os.path.exists(path) and os.path.isfile(path):
        performance = np.genfromtxt(path, delimiter=',')
        return performance
    else: print ('File Not Found Error: ' + path); return None;

class color:
    """color class for printing colored text"""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'