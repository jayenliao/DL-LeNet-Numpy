import pickle, sys
import cupy as np
import numpy
from collections import OrderedDict
from source.layers import *
from source.utils import *
from source.utils import cp2np

# The main model

class LeNet5:
    '''
    conv (W1, b1) -> act. fun. -> conv (W2, b2) -> act. fun. -> ... -> conv [num_conv_layers - 1] -> act. fun. ->
    affine [num_conv_layers + 1] -> act. fun. -> affine [num_conv_layers + 2] -> last softmax
    '''
    def __init__(self, model_name:str, input_dim:tuple, num_conv_layers:int, conv_params:dict, pooling_size:int, pooling_stride:int, hidden_sizes:list, hidden_act:str, output_size:int):
        # Check types of arguments
        assert len(input_dim) == 3
        assert len(hidden_sizes) >= 2
        for k in ['filter_num', 'filter_size', 'stride', 'pad']:
            assert k in conv_params.keys()
        
        # Initialize parameters
        hidden_size_0, hidden_size_1 = hidden_sizes[0], hidden_sizes[1]
        pre_node_nums = []
        pre_channel_num = input_dim[0]
        
        #print('pre_channel_num', pre_channel_num)

        pre_node_nums.append(pre_channel_num * conv_params['filter_size'][0] * conv_params['filter_size'][0]) 
        for l in range(1, num_conv_layers):
            filter_num = conv_params['filter_num'][l]
            filter_size = conv_params['filter_size'][l+1] if l < num_conv_layers - 1 else conv_params['filter_size'][l] # The last layer
            pre_node_nums.append(filter_num * filter_size * filter_size)  
        
        if conv_params['filter_size'][num_conv_layers - 1] == 5:
            L0 = 58 if input_dim[1] == 64 else 122
            last_conv_dim = L0 - 3 * (num_conv_layers - 2)
        elif conv_params['filter_size'][num_conv_layers - 1] == 3:
            L0 = 62 if input_dim[1] == 64 else 126
            last_conv_dim = L0 - 1 * (num_conv_layers - 2)
        else:
            last_conv_dim = 31 - conv_params['filter_size'][num_conv_layers - 1] + 3
            
        pre_node_nums.append(conv_params['filter_num'][num_conv_layers - 1] * last_conv_dim * last_conv_dim)
        pre_node_nums.append(hidden_size_0)
        pre_node_nums.append(hidden_size_1)
        pre_node_nums = np.array(pre_node_nums)
        #pre_node_nums = np.array([3*3*3 = 27, 16*3*3 = 144, 16*4*31*31 = 61504, hidden_size_0, hidden_size_1])
        #pre_node_nums = np.array([1*3*3 =  9, 16*3*3 = 144, 64*4*4 = 1024, hidden_size])
        #pre_node_nums = np.array([1*3*3 =  9, 16*3*3 = 144, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        
        print('pre_node_nums', pre_node_nums)
        # Recommended initial value when using ReLU, reference:  <----------- check
        weight_init_scales = np.sqrt(2.0 / pre_node_nums) if hidden_act.lower() == 'relu' else np.random.random(len(pre_node_nums))
        
        self.params = {}
        for l in range(num_conv_layers):
            Cin = conv_params['filter_num'][l]
            F = conv_params['filter_size'][l]
            #self.params['W' + str(l+1)] = weight_init_scales[l] * np.random.randn(conv_params['filter_num'][l], pre_channel_num, conv_params['filter_size'][l], conv_params['filter_size'][l])
            self.params['W' + str(l+1)] = weight_init_scales[l] * np.random.randn(Cin, pre_channel_num, F, F)
            #self.params['b' + str(l+1)] = np.zeros(conv_params['filter_num'][l])
            self.params['b' + str(l+1)] = np.zeros(Cin)
            pre_channel_num = conv_params['filter_num'][l]
        
        self.params['W' + str(num_conv_layers+1)] = weight_init_scales[num_conv_layers] * np.random.randn(conv_params['filter_num'][num_conv_layers-1] * last_conv_dim * last_conv_dim, hidden_size_0)
        self.params['b' + str(num_conv_layers+1)] = np.zeros(hidden_size_0)
        self.params['W' + str(num_conv_layers+2)] = weight_init_scales[num_conv_layers+1] * np.random.randn(hidden_size_0, hidden_size_1)
        self.params['b' + str(num_conv_layers+2)] = np.zeros(hidden_size_1)
        self.params['W' + str(num_conv_layers+3)] = weight_init_scales[num_conv_layers+2] * np.random.randn(hidden_size_1, output_size)
        self.params['b' + str(num_conv_layers+3)] = np.zeros(output_size)

        # Construct layers in order
        self.layers = []
        self.layer_names = []

        #if model_name.lower() == 'lenet5':
        #    num_conv_layers = 2
        for l in range(num_conv_layers):
            #print(str(l+1), self.params['W' + str(l+1)].shape)
            self.layers.append(Convolution(self.params['W' + str(l+1)], self.params['b' + str(l+1)], conv_params['stride'][l], conv_params['pad'][l]))
            self.layer_names.append('conv')
            self.layers.append(set_activation_function(hidden_act))
            self.layer_names.append('act. fun.')
            self.layers.append(Pooling(pool_h=pooling_size, pool_w=pooling_size, stride=pooling_stride))
            self.layer_names.append('maxpooling')

        #print('W' + str(num_conv_layers+1), self.params['W' + str(num_conv_layers+1)].shape)
        self.layers.append(Affine(self.params['W' + str(num_conv_layers+1)], self.params['b' + str(num_conv_layers+1)]))
        self.layer_names.append('linear')
        self.layers.append(set_activation_function(hidden_act))
        self.layer_names.append('act. fun.')

        self.layers.append(Affine(self.params['W' + str(num_conv_layers+2)], self.params['b' + str(num_conv_layers+2)]))
        self.layer_names.append('linear')
        self.layers.append(set_activation_function(hidden_act))
        self.layer_names.append('act. fun.')

        #print('W' + str(num_conv_layers+3), self.params['W' + str(num_conv_layers+3)].shape)
        self.layers.append(Affine(self.params['W' + str(num_conv_layers+3)], self.params['b' + str(num_conv_layers+3)]))
        self.layer_names.append('linear')

        # The last layer
        self.last_layer = SoftmaxWithCrossEntropyLoss()
        
    def predict(self, x, train_flg=False):
        for layer in self.layers:
            try:
                x = layer.forward(x)
                #print(layer, x.shape)
            except:
                print('ERROR!', layer, x.shape)
                raise
            '''
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
            '''
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy_score(self, x, yt, top=1):
        if yt.ndim != 1:
            yt = np.argmax(yt, axis=1)
        y = self.predict(x)
        if top == 1:
            y = np.argmax(y, axis=1)
            acc = np.array(y == yt).mean()
        else:
            y = np.argsort(y, axis=1)[:,-top:]
            lst = []
            for i in range(len(yt)):
                lst.append(yt[i] in y[i,:])
            acc = np.array(lst).mean()
        return cp2np(acc)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # store values
        grads, params_i = {}, 1
        for layer_i, layer_name in enumerate(self.layer_names):
            if layer_name in ['conv', 'linear']:
                grads['W' + str(params_i)] = self.layers[layer_i].dW
                grads['b' + str(params_i)] = self.layers[layer_i].db
                params_i += 1
        return grads

    def save_params(self, PATH, fn='params.pkl'):
        params = {}
        for k, v in self.params.items():
            params[k] = v
        PATH = PATH if PATH[-1] == '/' else PATH+'/'
        with open(PATH + fn, 'wb') as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    def load_params(self, PATH, fn='params.pkl'):
        params = load_file(PATH, fn)
        for k, v in params.items():
            self.params[k] = v
        params_i = 1
        for layer_i, layer_name in enumerate(self.layer_names):
            if layer_name in ['conv', 'linear']:
                self.layers[layer_i].W = self.params['W' + str(params_i)].dW
                self.layers[layer_i].b = self.params['b' + str(params_i)].db
                params_i += 1

                
# For debugging

class DeepConvNet:
    def __init__(self, input_dim,
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 #conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 #conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 #conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=50):

        pre_node_nums = np.array([3*3*3, 16*3*3, 16*3*3, 32*32*32, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums) 
        
        print(pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W4'] = weight_init_scales[3] * np.random.randn(32*32*32, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = weight_init_scales[4] * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        self.layers, self.layer_names = [], []
        
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],  # 0
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layer_names.append('conv')
        
        self.layers.append(ReLU())                                            # 1
        self.layer_names.append('relu')
        
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],  # 2
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layer_names.append('conv')
        
        self.layers.append(ReLU())                                            # 3
        self.layer_names.append('relu')
        
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))             # 4
        self.layer_names.append('pool')
        
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],  # 5
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layer_names.append('conv')
        
        self.layers.append(ReLU())                                            # 6
        self.layer_names.append('relu')
        
        '''
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],  # 7
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(ReLU())                                            # 8
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))             # 9
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],  # 10
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(ReLU())                                            # 11
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],  # 12
                           conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(ReLU())                                            # 13
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))             # 14
        '''
        self.layers.append(Affine(self.params['W4'], self.params['b4']))      # 15
        self.layer_names.append('linear')
        
        self.layers.append(ReLU())                                            # 16
        self.layer_names.append('relu')
        
        self.layers.append(Affine(self.params['W5'], self.params['b5']))      # 17
        self.layer_names.append('linear')
        
        self.last_layer = SoftmaxWithCrossEntropyLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy_score(self, x, yt, top=1):
        if yt.ndim != 1:
            yt = np.argmax(yt, axis=1)
        y = self.predict(x)
        if top == 1:
            y = np.argmax(y, axis=1)
            acc = np.array(y == yt).mean()
        else:
            y = np.argsort(y, axis=1)[:,-top:]
            lst = []
            for i in range(len(yt)):
                lst.append(yt[i] in y[i,:])
            acc = np.array(lst).mean()
        return cp2np(acc)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        #for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 17)):
        grads, params_i = {}, 1
        for layer_i, layer_name in enumerate(self.layer_names):
            if layer_name in ['conv', 'linear']:
                grads['W' + str(params_i)] = self.layers[layer_i].dW
                grads['b' + str(params_i)] = self.layers[layer_i].db
                params_i += 1
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 17)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
