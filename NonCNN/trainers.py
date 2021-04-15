import os, sys, time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from NonCNN.utils import print_accuracy, smooth_curve
from NonCNN.optimizers import *
from NonCNN.models import *
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def set_optimizer(optimizer_name: str):
    if optimizer_name == 'SGD':
        optimizer = SGD()
    elif optimizer_name == 'Momentum':
        optimizer = Momentum()
    elif optimizer_name == 'AdaGrad':
        optimizer = AdaGrad()
    elif optimizer_name == 'Adam':
        optimizer = Adam()
    else:
        sys.exit('ERROR! The given optimizer name \'' + optimizer_name + '\' is not available!')
    return optimizer

class NonNNtrainer:
    def __init__(self, X_tr, y_tr, feature_type:str, model_:str, n_jobs:int, random_state=4028):
        self.trained = False
        self.feature_type = feature_type
        self.model_ = model_
        self.X_tr = X_tr
        self.y_tr = y_tr
        if self.model_.lower() == 'rf':
            self.model = RandomForestClassifier(random_state=random_state)
        elif self.model_.lower() == 'xgb':
            self.model = XGBClassifier(random_state=random_state, n_jobs=n_jobs)
    
    def train(self, save_model=True, savePATH='./output/', pretrained_model=''):
        savePATH = savePATH if savePATH[-1] == '/' else savePATH + '/'
        savePATH += self.model + '/'
        try:
            os.makedirs(savePATH)
        except FileExistsError:
            pass

        if pretrained_model == '':
            print('Training a new', self.model_, 'model ...')
            t0 = time.time()
            self.model.fit(self.X_tr, self.y_tr)
            self.trained = True
            tdiff = time.time() - t0
            print('Finish training! Time cost: %8.2f s' % tdiff)
            
            dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S') 
            self.fn = savePATH + dt + '_' + self.feature_type + '_' + self.model_ + '.pkl'
            if save_model:
                try:
                    os.makedirs(savePATH)
                except:
                    pass
                with open(self.fn, 'wb') as f:
                    pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)
                print('The model is saved as', self.fn)
        else:
            if pretrained_model.split('.')[-1] != 'pkl':
                pretrained_model = pretrained_model.split('.')[0] + '.pkl'
            self.fn = pretrained_model if '/' in pretrained_model else savePATH + pretrained_model
            print('Loading the pretrained', self.model_, 'model ...', end='  ')
            with open(fn, 'rb') as f:
                self.model = pickle.load(f)
            self.trained = True
            print('Done!')

    def evaluate(self, X_va=[], y_va=[]):
        print('Training performance:', end='  ')
        acc1, acc5 = print_accuracy(self.y_tr, self.model.predict_proba(self.X_tr))
        print(f'Top-1 accuracy={acc1:.4f}, Top-5 accuracy={acc5:.4f}')
        d_accuracy = {'top-1': [acc1], 'top-5': [acc5]}
        if len(X_va) + len(y_va) > 0:
            print('Evaluation performance', end='  ')
            acc1, acc5 = print_accuracy(y_va, self.model.predict_proba(X_va))
            print(f'Top-1 accuracy={acc1:.4f}, Top-5 accuracy={acc5:.4f}')
            d_accuracy['top-1'].append(acc1)
            d_accuracy['top-5'].append(acc5)
            pd.DataFrame(d_accuracy, index=['train', 'val']).to_csv(self.fn.replace('.pkl', '.csv'))

class NNtrainer:
    def __init__(self, X_tr, y_tr, X_te, y_te, feature_type:str, model:str, hidden_size:int, hidden_layer_act:str, output_layer_act:str, optimizer:str, lr:float, random_state:int):
        self.trained = False
        self.random_state = random_state
        self.feature_type = feature_type
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te
        self.y_te = y_te
        self.train_size = X_tr.shape[0]
        self.input_size = X_tr.shape[1]
        self.output_size = y_tr.shape[1]
        self.hidden_size = hidden_size
        self.lr = lr
        self.model_ = model
        print('\n---> Model:', model, '\nlr =', self.lr, end='| ')

        if self.model_ == 'NaivePerceptron':
            self.model = NaivePerceptron(self.input_size, self.output_size)
            print('\n\n')
        else:
            self.train_loss = []
            self.optimizer = set_optimizer(optimizer)
            if self.model_ == 'OneLayerPerceptron':
                self.model = OneLayerPerceptron(self.input_size, self.output_size, output_layer_act)
                print(' Optimizer:', optimizer, '| Act. fun. of the output layer:', output_layer_act, '| \n')
            elif self.model_ in ['TwoLayerPerceptron', 'TwoLayerNet']:
                self.model = TwoLayerNet(self.input_size, self.hidden_size, self.output_size, hidden_layer_act, output_layer_act)
                print(' Optimizer:', optimizer, '| Hidden size:', hidden_size, '| Act. fun. of the hidden layer:', hidden_layer_act, '| Act. fun. of the output layer:', output_layer_act, '| \n')
            else:
                sys.exit('ERROR! The given model name \'' + self.model_ + '\' is not available!')

    def evaluation_when_training(self, epoch:int, print_result_per_epochs:int, X_va, y_va):
        # Compute the top-1 and top-5 accuracy scores on the training set
        acc1 = self.model.accuracy_score(self.X_batch, self.y_batch, top=1)
        self.accuracy_tr['top-1'].append(acc1)
        acc5 = self.model.accuracy_score(self.X_batch, self.y_batch, top=5)
        self.accuracy_tr['top-5'].append(acc5)
            
        # Print the training accuracy scores once per given no. of epochs
        if epoch % print_result_per_epochs == 0:
            print(f'Epoch {epoch:4d}')
            if self.model_ != 'NaivePerceptron':
                print(f'Training loss={self.loss_:.4f}')
            print(f'  Training accuracy: {acc1:.4f} (top-1) {acc5:.4f} (top-5)')

        # If the evaluation set is given, compute the top-1 and top-5 accuracy scores on it
        if self.evaluation:
            acc1 = self.model.accuracy_score(X_va, y_va, top=1)
            self.accuracy_va['top-1'].append(acc1)
            acc5 = self.model.accuracy_score(X_va, y_va, top=5)
            self.accuracy_va['top-5'].append(acc5)
                
            # Find the best performance
            if acc1 > self.best_acc['best_acc1']:
                self.best_acc['best_acc1'] = acc1
                self.best_acc['best_acc1_epoch'] = epoch
            if acc5 > self.best_acc['best_acc5']:
                self.best_acc['best_acc5'] = acc5
                self.best_acc['best_acc5_epoch'] = epoch
            if (acc1 + acc5) / 2 > self.best_acc['best_acc_mean']:
                self.best_acc['best_acc_mean'] = (acc1 + acc5) / 2
                self.best_acc['best_acc_mean_epoch'] = epoch
                
            # Print the validation accuracy scores once per given no. of epochs
            if epoch % print_result_per_epochs == 0:
                print(f'Validation accuracy: {acc1:.4f} (top-1) {acc5:.4f} (top-5)')

    def train(self, epochs:int, batch_size:int, save_trainer=True, save_model=True, savePATH='./output/', X_va=[], y_va=[], print_result_per_epochs=10, pretrained_model='', verbose=False):
        self.epochs = epochs
        self.evaluation = True if len(X_va) + len(y_va) > 1 else False
        self.savePATH = savePATH if savePATH[-1] == '/' else savePATH + '/'

        # Load the pretrained model or train a new model
        if pretrained_model == '':
            print('Training a new model ...', end='  ')
            self.accuracy_tr = {'top-1': [], 'top-5': []}
            self.accuracy_va = {'top-1': [], 'top-5': []}
            self.accuracy_te = {'Top-1': 0, 'Top-5': 0}
            self.best_acc = {
                'best_acc1': 0, 'best_acc1_epoch': 0, 
                'best_acc5': 0, 'best_acc5_epoch': 0,
                'best_acc_mean': 0, 'best_acc_mean_epoch': 0
            }
        else:
            print('Loading the pretrained model ...')
            fn = self.savePATH + pretrained_model
            with open(fn, 'rb') as f:
                self.model = pickle.load(f)
            fn = self.savePATH + fn.replace('Model', 'Accuracy').replace('.pkl', '.txt')
            arr = np.loadtxt(fn)
            self.accuracy_tr = {'top-1': list(arr[:,0]), 'top-5': list(arr[:,1])}
            if arr.shape[1] == 4:
                self.accuracy_va = {'top-1': list(arr[:,2]), 'top-5': list(arr[:,3])}
            fn = fn.replace('Acc', 'BestAcc')
            self.best_acc = pd.read_csv(fn, index_col=0).T.to_dict('records')[0]
            print('Keep training the model ...', end='  ')
        print(f'(Batch size = {batch_size}, No. of epochs = {epochs})')
        
        # Training
        time_cost = []
        t0 = time.time()
        np.random.seed(self.random_state)
        for epoch in range(1, self.epochs + 1):
            tEpoch = time.time()
            if self.model_ != 'NaivePerceptron':
                idx_list = np.arange(self.train_size)
                np.random.shuffle(idx_list) # Get the batch loader
                batch_temp = 0
                loader = list(range(batch_size, self.train_size, batch_size))
                if not verbose:
                    loader = tqdm(loader)
                for batch in loader:
                    if self.train_size - batch > batch_size:
                        batch_mask = idx_list[batch_temp:batch]
                    else:  # The last batch
                        batch_mask = idx_list[batch:]
                    self.X_batch = self.X_tr[batch_mask]
                    self.y_batch = self.y_tr[batch_mask]
                    grads = self.model.gradient(self.X_batch, self.y_batch)
                    self.optimizer.update(self.model.params, grads) # Update model parameters
                    self.loss_ = self.model.loss(self.X_batch, self.y_batch)
                    self.train_loss.append(self.loss_)
                    batch_temp = batch
            else:
                y = self.model.activation(self.X_tr)
                self.model.weights[1:,:] += self.lr * np.dot(self.X_tr.T, (self.y_tr - y))
                self.model.weights[0,:]  += self.lr * (self.y_tr - y).sum(axis=0)
            self.evaluation_when_training(epoch, print_result_per_epochs, X_va, y_va)
            self.accuracy_te['Top-1'] = self.model.accuracy_score(self.X_te, self.y_te, top=1)
            self.accuracy_te['Top-5'] = self.model.accuracy_score(self.X_te, self.y_te, top=5)
            tStep = time.time()
            time_cost.append(tStep - tEpoch)

        self.trained = True
        tdiff = time.time() - t0
        print('\nFinish training! Time cost: %.2f s' % tdiff)
        
        # Save the model performance
        self.dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        self.folder_name = self.savePATH + self.dt + '_' + self.model_ + '_' + self.feature_type + '_bs=' + str(batch_size) + '_epochs=' + str(epochs)
        if self.model_ == 'TwoLayerNet':
            self.folder_name += '_hs=' + str(self.hidden_size)
        self.folder_name += '/'
        try:
            os.makedirs(self.folder_name)
        except FileExistsError:
            pass
        
        print('Model performances are saved as the following files:')
        arr = [self.accuracy_tr['top-1'], self.accuracy_tr['top-5']]
        if self.evaluation:
            arr += [self.accuracy_va['top-1'], self.accuracy_va['top-5']] 
        self.fn = self.folder_name + 'Accuracy.txt'
        np.savetxt(self.fn, arr)
        print('-->', self.fn)

        fn = self.fn.replace('Acc', 'TestAcc')
        pd.DataFrame(self.accuracy_te, index=['score']).T.to_csv(fn)
        print('-->', fn)

        fn = self.fn.replace('Acc', 'BestAcc')
        pd.DataFrame(self.best_acc, index=[0]).T.to_csv(fn)
        print('-->', fn)

        fn = self.fn.replace('Accuracy', 'TimeCost')
        np.savetxt(fn, np.array(time_cost))
        print('-->', fn)

        # Save the trained trainer or model
        if save_model and not save_trainer:  # If the trainer has been saved, then we do not save the model again.
            fn = self.fn.replace('Accuracy', 'Model').replace('.txt', '.pkl')
            with open(fn, 'wb') as f:
                pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)
            print('\nThe model is saved as', fn)

    def plot_training(self, type_:str, figsize=(8, 6), save_plot=True):
        plt.figure(figsize=figsize)
        if type_ == 'loss':
            x = np.arange(len(self.train_loss))
            plt.plot(x, smooth_curve(self.train_loss))
            plt.title('Plot of Training Loss of ' + self.model_)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
        elif type_ == 'accuracy':
            x = np.arange(1, self.epochs+1)
            for k in self.accuracy_tr.keys():
                plt.plot(x, self.accuracy_tr[k], label=k+' accuracy (train)')
                if self.evaluation:
                    plt.plot(x, self.accuracy_va[k], label=k+' accuracy (val)')
            plt.ylim(0, 1)
            plt.title('Plot of Accuracy During Training of ' + self.model_)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        plt.grid()
        #plt.show()
        if save_plot:
            fn = self.fn.replace('Accuracy', type_).replace('.txt', '.png')
            plt.savefig(fn)
            print('The', type_, 'plot is saved as', fn)