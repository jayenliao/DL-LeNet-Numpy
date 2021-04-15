import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from lenet_torch.model import LeNet

class Trainer:
    def __init__(self, X_tr, y_tr, X_te, y_te, device, model_name:str, model_hyper_param:dict, optimizer:str, lr:float, epochs:int, batch_size:int, savePATH:str, save_model:bool, pretrained_model:str, verbose:bool, random_state=4028):
        self.model_name = model_name
        
        if model_name.lower() == 'lenet5':
            self.model = LeNet(**model_hyper_param)        
        elif model_name.lower() == 'improved_lenet5':
            # ...
            self.model = LeNet(**model_hyper_param)
        else:
            sys.exit('ERROR! The model name cannot be ' + model_name + '.')

        self.device = device
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te.to(self.device)
        self.y_te = y_te.to(self.device)
        self.model = self.model.to(device)
        #self.num_conv_layers = model_hyper_param['num_conv_layers']
        self.hidden_act = model_hyper_param['hidden_act']
        #self.hidden_sizes = model_hyper_param['hidden_sizes']
        #self.filter_size = model_hyper_param['filter_size']
        #self.pooling_size = model_hyper_param['pooling_size']
        self.output_size = model_hyper_param['output_size']
        self.train_size = self.Ｘ_tr.shape[0]
        self.batch_size = batch_size
        self.epochs = epochs
        self.savePATH = savePATH if savePATH[-1] == '/' else savePATH + '/'
        self.pretrained_model = pretrained_model
        self.random_state = random_state
        self.verbose = verbose
        self.train_loss = []
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def accuracy_score(self, y, yt, top=1):
        y = y.cpu().numpy()
        yt = yt.cpu().numpy()

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

    def evaluation_when_training(self, epoch:int, print_result_per_epochs:int, X_va, y_va):
        # Compute the top-1 and top-5 accuracy scores on the training set
        acc1 = self.accuracy_score(self.output, self.y_batch, top=1)
        self.accuracy_tr['Top-1'].append(acc1)
        acc5 = self.accuracy_score(self.output, self.y_batch, top=5)
        self.accuracy_tr['Top-5'].append(acc5)
            
        # Print the training accuracy scores once per given no. of epochs
        if epoch % print_result_per_epochs == 0:
            print(f'Epoch {epoch:4d}')
            print(f'  Training accuracy: {acc1:.4f} (top-1) {acc5:.4f} (top-5)')

        # If the evaluation set is given, compute the top-1 and top-5 accuracy scores on it
        if self.evaluation:
            acc1 = self.accuracy_score(self.model(X_va), y_va, top=1)
            self.accuracy_va['Top-1'].append(acc1)
            acc5 = self.accuracy_score(self.model(X_va), y_va, top=5)
            self.accuracy_va['Top-5'].append(acc5)
            
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


    def train_single_epoch(self):
        idx_list = np.arange(self.train_size)
        np.random.seed(self.random_state)
        np.random.shuffle(idx_list)         # Get the batch loader
        batch_temp = 0
        loader = list(range(self.batch_size, self.train_size, self.batch_size))
        for batch in tqdm(loader):
            if self.train_size - batch > self.batch_size:
                batch_mask = idx_list[batch_temp:batch]
            else: # The last batch
                batch_mask = idx_list[batch:]
            self.X_batch = self.X_tr[batch_mask].to(self.device)
            self.y_batch = self.y_tr[batch_mask].to(self.device)
            #print('X_batch', X_batch.shape)
            #print('y_batch', y_batch.shape)
            self.output = self.model(self.X_batch)
            loss = self.criterion(self.output, self.y_batch)
            self.train_loss.append(loss)
            batch_temp = batch
            #if batch_temp > loader[10000]:
            #    break

    def train(self, X_va=[], y_va=[], print_result_per_epochs=10, save_model=True):
        '''
        1. Initialize
        2. Load the pretrained model (if the pretrained model name is given) or train a new model
        3. Training for no. of epochs
        4. Report that the model have been finished training and 
           Save the model performances: acc_tr/acc_va, acc_te, best_acc, and time_cost
        '''
        
        # 1. Intialize
        self.evaluation = True if len(X_va) + len(y_va) > 1 else False
        if self.evaluation:
            X_va = X_va.to(self.device)
            y_va = y_va.to(self.device)

        # 2. Load the pretrained model or train a new model
        if self.pretrained_model == '':
            print('\nTraining a new model ...', end='  ')
            self.accuracy_tr = {'Top-1': [], 'Top-5': []}
            self.accuracy_va = {'Top-1': [], 'Top-5': []}
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
            self.accuracy_tr = {'Top-1': list(arr[:,0]), 'Top-5': list(arr[:,1])}
            if arr.shape[1] == 4:
                self.accuracy_va = {'Top-1': list(arr[:,2]), 'Top-5': list(arr[:,3])}
            fn = fn.replace('Acc', 'BestAcc')
            self.best_acc = pd.read_csv(fn, index_col=0).T.to_dict('records')[0]
            print('Keep training the model ...', end='  ')
        if not self.verbose:
            print(f'(Batch size = {self.batch_size}, No. of epochs = {self.epochs})')

        # 3. Training! 
        time_cost = []
        t0 = time.time()
        for epoch in range(1, self.epochs+1):
            tEpoch = time.time()
            self.train_single_epoch()
            self.evaluation_when_training(epoch, print_result_per_epochs, X_va, y_va)
            self.accuracy_te['Top-1'] = self.accuracy_score(self.model(self.X_te), self.y_te, top=1)
            self.accuracy_te['Top-5'] = self.accuracy_score(self.model(self.X_te), self.y_te, top=5)
            tStep = time.time()
            time_cost.append(tStep - tEpoch)

        # 4. Report that the model have been finished training and 
        #    Save the model performances: acc_tr/acc_va, acc_te, best_acc, and time_cost
        self.trained = True
        tdiff = time.time() - t0
        if not self.verbose:
            print('\nFinish training! Total time cost for %3d epochs: %.2f s' % (self.epochs, tdiff))
        arr = [self.accuracy_tr['Top-1'], self.accuracy_tr['Top-5']]
        if self.evaluation: arr += [self.accuracy_va['Top-1'], self.accuracy_va['Top-5']] 
        
        if not self.verbose:
            print('\nEvaluate the trained model on the testing set ...')
            print_accuracy(self.y_te, self.model(self.X_te))

        if not self.verbose:
            print('Model performances are saved as the following files:')
        self.dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        #self.folder_name = self.savePATH + self.dt + '_' + str(self.num_conv_layers) + '_' + self.model_name + '_' + self.hidden_act + '_fs=' + str(self.filter_size) + '_bs=' + str(self.batch_size) + '_epochs=' + str(self.epochs) + '/'
        self.folder_name = self.savePATH + self.dt + '_' + str(self.num_conv_layers) + '_' + self.model_name + '_' + self.hidden_act + '_fs=' + str(self.filter_size) + '_bs=' + str(self.batch_size) + '_epochs=' + str(self.epochs) + '/'
        try:
            os.makedirs(self.folder_name)
        except FileExistsError:
            pass
        
        self.fn = self.folder_name + 'Accuracy.txt'
        numpy.savetxt(self.fn, arr)
        if not self.verbose:
            print('-->', self.fn)

        fn = self.fn.replace('Acc', 'TestAcc')
        pd.DataFrame(self.accuracy_te, index=['score']).T.to_csv(fn)
        if not self.verbose:
            print('-->', fn)

        fn = self.fn.replace('Acc', 'BestAcc')
        pd.DataFrame(self.best_acc, index=[0]).T.to_csv(fn)
        if not self.verbose:
            print('-->', fn)

        fn = self.fn.replace('Accuracy', 'TimeCost')
        numpy.savetxt(fn, numpy.array(time_cost))
        if not self.verbose:
            print('-->', fn)

        # 5. Save the trained model
        if save_model:
            print('Saving model ....')
            fn = self.fn.replace('Accuracy', 'Model').replace('.txt', '.pkl')
            with open(fn, 'wb') as f:
                pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)
            if not self.verbose:
                print('\nThe model is saved as', fn)

    def plot_training(self, type_:str, figsize=(8, 6), save_plot=True):
        plt.figure(figsize=figsize)
        if type_ == 'loss':
            x = numpy.arange(len(self.train_loss))
            plt.plot(x, smooth_curve(self.train_loss))
            plt.title('Plot of Training Loss of ' + self.model_name)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
        elif type_ == 'accuracy':
            x = numpy.arange(1, self.epochs+1)
            for k in self.accuracy_tr.keys():
                plt.plot(x, self.accuracy_tr[k], label=k+' accuracy (train)')
                if self.evaluation:
                    plt.plot(x, self.accuracy_va[k], label=k+' accuracy (val)')
            plt.ylim(0, 1)
            plt.title('Plot of Accuracy During Training of ' + self.model_name)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        plt.grid()
        #plt.show()
        if save_plot:
            fn = self.fn.replace('Accuracy', type_).replace('.txt', '.png')
            plt.savefig(fn)
            print('The', type_, 'plot is saved as', fn)