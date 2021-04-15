import numpy
import cupy as np
import numpy as np
import pandas as pd
import cv2
from source.layers import *

def load_original_data(subset:str, PATH='./', fn_type='.txt', sep=' ', header=None):
    df = pd.read_table(PATH+subset+fn_type, sep=sep, header=header)
    fn_list, y = list(df.iloc[:,0]), np.array(df.iloc[:,1])
    return fn_list, y

def load_images(subset:str, PATH='./', fn_type='.txt', sep=' ', header=None):
    fn_list, y = load_original_data(subset, PATH, fn_type, sep, header)
    img_list = []
    for fn in fn_list:
        img_list.append(cv2.imread(fn))
    return img_list, fn_list, y

def img_resize(img_list:list, resize:tuple) -> np.array:
    img_list_resized = []
    for img in img_list:
        img_list_resized.append(cv2.resize(img, resize))
    return np.array(img_list_resized)

def get_resized_data(subset:str, PATH:str, resize:tuple, fn_type='.txt', sep=' ', header=None):
    try:
        print('Loading the resized', subset, 'images ...', end=' ')
        X = np.load(PATH + 'X_' + subset[:2] + '_' + str(resize[0]) + '.npy', allow_pickle=True)
        y = np.load(PATH + 'y_' + subset[:2] + '.npy', allow_pickle=True)
    except:
        print('Failed! QAQ')
        print('Loading the original', subset, 'images ...', end=' ')
        img_list, fn_list, y = load_images(subset, PATH, fn_type, sep, header)
        X = img_resize(img_list, resize)
        np.save(PATH + 'X_' + subset[:2] + '_' + str(resize[0]) + '.npy', X)
        np.save(PATH + 'y_' + subset[:2] + '.npy', y)
    print('Done!')
    return X, y

def resize_channel(X:np.array):
    if X.shape[1] > 3:    # (N, 64, 64, 3) -> (N, 3, 64, 64)
        X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])
    elif X.shape[-1] > 3: # (N, 3, 64, 64) -> (N, 64, 64, 3)
        X = X.reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[1])
    return X

def one_hot_transformation(y):
    k = len(np.unique(y))
    return np.eye(k)[y]

def check_dim(img_list):
    l = []
    for arr in img_list:
        l.append([arr.shape[0], arr.shape[1], arr.shape[2]])
    return np.array(l)

def accuracy_score_(yt, y, top=1):
    if yt.ndim != 1:
        yt = np.argmax(yt, axis=1)
    if top == 1:
        y = np.argmax(y, axis=1)
        acc = np.array(y == yt).mean()
    else:
        y = np.argsort(y, axis=1)[:,-top:]
        lst = []
        for i in range(len(yt)):
            lst.append(yt[i] in y[i,:])
        acc = np.array(lst).mean()
    return acc

def cp2np(arr):
    if type(arr) == np.core.core.ndarray:  # We import cupy as np
        arr = arr.get()
    return arr

def print_accuracy(yt, yp, print_results=True):
    acc1 = cp2np(accuracy_score_(yt, yp, top=1))
    acc5 = cp2np(accuracy_score_(yt, yp, top=5))
    if print_results:
        print(f'Top-1 accuracy={acc1:.4f}, Top-5 accuracy={acc5:.4f}')
    else:
        return acc1, acc5

def smooth_curve(x): 
    # Reference: http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    window_len = 11
    s = numpy.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = numpy.kaiser(window_len, 2)
    y = numpy.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

def set_activation_function(function_name:str):
    if function_name.lower() == 'sigmoid':
        return Sigmoid()
    elif function_name.lower() == 'relu':
        return ReLU()
    elif function_name.lower() == 'tanh':
        return tanh()
    else:
        sys.exit('ERROR! Get an unrecognized name of activation function.')
