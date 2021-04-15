import cupy as np
#import numpy as np
from source.utils import *

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1. /(1 + np.exp(-x))

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
    return -np.array(np.log(y[np.arange(batch_size), yt] + 1e-7)).mean()

def cross_entropy_error(y, yt):
    if y.ndim == 1:
        yt = yt.reshape(1, yt.size)
        y = y.reshape(1, y.size)
    if yt.size == y.size:
        yt = yt.argmax(axis=1)     
    batch_size = y.shape[0]
    #print('y', y.shape)
    #print('yt', yt.shape)
    return -np.array(np.log(y[np.arange(batch_size), yt] + 1e-7)).mean()

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

'''
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
'''

def im2col(input_data, filter_h:int, filter_w:int, stride=1, pad=0):
    '''
    Given
        1. input_data: input data consisting of a 4-D array of (#observations, #channels, height, width)
        2. filter_h: filter height
        3. filter_w: filter width
        4. stride: stride
        5. pad: padding
    Return
        col: 2-D array
    '''
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape:tuple, filter_h:int, filter_w:int, stride=1, pad=0):
    '''
    Given
        1. input_shape: shape of input data (#observations, #channels, height, width)
        2. filter_h: filter height
        3. filter_w: filter width
        4. stride: stride
        5. pad: padding
    Return
        im: image data
    '''
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class tanh:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.tanh(x)

    def backward(self, out):
        dout = self.cache
        dx = dout * (1 - np.tanh(out)**2)
        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1. - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        try:
            out = np.dot(self.x, self.W) + self.b
        except:
            print(self.x.shape, self.W.shape)
            raise
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape) 
        return dx

class SoftmaxWithCrossEntropyLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, yt):
        self.yt = yt
        self.y = softmax(x)
        #print(self.y.shape)
        #print(self.yt.shape)
        self.loss = cross_entropy_error(self.y, self.yt)
        return self.loss

    def backward(self, dout):
        batch_size = self.yt.shape[0]
        if self.yt.size == self.y.size: # deal with one-hot-encoding y
            dx = (self.y - self.yt) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.yt] -= 1
            dx = dx / batch_size
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W     # weights
        self.b = b     # bias
        self.dW = None # gradient of weights
        self.db = None # gradient of bias
        self.stride = stride
        self.pad = pad
        self.x = None   
        self.col = None
        self.col_W = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        #print(col.shape, col_W.shape)

        try:
            out = np.dot(col, col_W) + self.b
        except:
            print(col.shape, col_W.shape)
            raise
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
