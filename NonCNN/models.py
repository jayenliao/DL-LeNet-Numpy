import numpy as np
from NonCNN.layers import *
from NonCNN.utils import numerical_gradient
from collections import OrderedDict

class NaivePerceptron:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((input_size+1, output_size)) # Treat the first row as the bias term
           
    def predict(self, x):
        return np.dot(x, self.weights[1:,:]) + self.weights[0,:]
    
    def activation(self, x):
        return (self.predict(x) > 0)*1  # Pass the message if the summation is positive

    def accuracy_score(self, x, yt, top):
        y = self.predict(x)
        return accuracy_score_(yt, y, top)


class OneLayerPerceptron:
    def __init__(self, input_size:int, output_size:int, output_layer_act:str, weight_init_std=.01):
        # Initialize parameters
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)

        # Construct layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        if output_layer_act == 'ReLU':
            self.output_layer = ReLU()
        elif output_layer_act == 'Sigmoid':
            self.output_layer = Sigmoid()
        elif output_layer_act == 'SoftmaxWithCrossEntropyLoss':
            self.output_layer = SoftmaxWithCrossEntropyLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.output_layer.forward(y, t)
    
    def accuracy_score(self, x, yt, top=1):
        if yt.ndim != 1:
            yt = np.argmax(yt, axis=1)
        y = self.predict(x)
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
        
    def numerical_gradient_(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = _numerical_gradient_1d(loss_W, self.params['W1'])
        grads['b1'] = _numerical_gradient_1d(loss_W, self.params['b1'])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.output_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # store values
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db

        return grads


class TwoLayerNet:
    def __init__(self, input_size:int, hidden_size:int, output_size:int, hidden_layer_act:str, output_layer_act:str, weight_init_std=.01):
        # Initialize parameters
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # Construct layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])

        if hidden_layer_act == 'ReLU':
            self.layers['Relu1'] = ReLU()
        elif hidden_layer_act == 'Sigmoid':
            self.layers['Sigmoid1'] = Sigmoid()
        elif hidden_layer_act == 'tanh':
            self.layers['tanh'] = tanh()

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        if output_layer_act == 'ReLU':
            self.output_layer = ReLU()
        elif output_layer_act == 'Sigmoid':
            self.output_layer = Sigmoid()
        elif output_layer_act == 'SoftmaxWithCrossEntropyLoss':
            self.output_layer = SoftmaxWithCrossEntropyLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.output_layer.forward(y, t)
    
    def accuracy_score(self, x, yt, top=1):
        if yt.ndim != 1:
            yt = np.argmax(yt, axis=1)
        y = self.predict(x)
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
        
    def numerical_gradient_(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = _numerical_gradient_1d(loss_W, self.params['W1'])
        grads['b1'] = _numerical_gradient_1d(loss_W, self.params['b1'])
        grads['W2'] = _numerical_gradient_1d(loss_W, self.params['W2'])
        grads['b2'] = _numerical_gradient_1d(loss_W, self.params['b2'])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.output_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # store values
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads