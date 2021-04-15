import numpy as np
from NonCNN.utils import *

class tanh:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        self.out = np.tanh(self.cache)
        return self.out

    def backward(self, dout):
        dout = self.cache
        dx = dout * (1 - np.tanh(dout)**2)
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
        out = np.dot(self.x, self.W) + self.b
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
