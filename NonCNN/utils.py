import numpy as np
import pandas as pd
import cv2

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

def get_resized_data(subset:str, PATH:str, fn_type='.txt', sep=' ', header=None):
    try:
        print('Loading the resized', subset, 'images ...')
        X = np.load(PATH + 'X_' + subset[:2] + '.npy', allow_pickle=True)
        y = np.load(PATH + 'y_' + subset[:2] + '.npy', allow_pickle=True)
    except:
        print('Loadig the original', subset, 'images ....')
        img_list, fn_list, y = load_images(subset, PATH, fn_type, sep, header)
        X = img_resize(img_list, args.resize)
        np.save(PATH + 'X_' + subset[:2] + '.npy', X)
        np.save(PATH + 'y_' + subset[:2] + '.npy', y)
    return X, y

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
        acc = np.mean(y == yt)
    else:
        y = np.argsort(y, axis=1)[:,-top:]
        lst = []
        for i in range(len(yt)):
            lst.append(yt[i] in y[i,:])
        acc = np.mean(lst)
    return acc

def print_accuracy(yt, yp, print_results=True):
    acc1 = accuracy_score_(yt, yp, top=1)
    acc5 = accuracy_score_(yt, yp, top=5)
    if print_results:
        print(f'Top-1 accuracy={acc1:.4f}, Top-5 accuracy={acc5:.4f}')
    else:
        return acc1, acc5

def smooth_curve(x): 
    # Reference: http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad
    
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def cross_entropy_error(y, yt):
    if y.ndim == 1:
        yt = yt.reshape(1, yt.size)
        y = y.reshape(1, y.size)
    if yt.size == y.size:
        yt = yt.argmax(axis=1)     
    batch_size = y.shape[0]
    return -np.mean(np.log(y[np.arange(batch_size), yt] + 1e-7))

def _numerical_gradient_1d(f, x):
    h = 1e-4 
    grad = np.zeros_like(x)
    for i in range(x.size):
        temp = x[i]
        x[i] = float(temp) + h
        fxh1 = f(x) # f(x+h)
        x[i] = temp - h 
        fxh2 = f(x) # f(x-h)
        grad[i] = (fxh1 - fxh2) / (h*2)
        x[i] = temp 
    return grad

def numerical_gradient(f, x):
    h = 1e-4  # to avoid zero
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        temp = x[i]
        x[i] = temp + h
        fxh1 = f(x) # f(x+h)
        x[i] = temp - h 
        fxh2 = f(x) # f(x-h)
        grad[i] = (fxh1 - fxh2) / (h*2)
        x[i] = temp
        it.iternext()   
    return grad
