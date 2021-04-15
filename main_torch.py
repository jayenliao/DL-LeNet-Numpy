'''
Deep Learning - HW3: LeNet
Jay Liao (re6094028@gs.ncku.edu.tw)
'''

import time, os
import cupy as np
import numpy as np
import pandas as pd
import torch
from lenet_torch.args import init_arguments
from lenet_torch.utils import *
from lenet_torch.trainer import Trainer

def main(args, return_trainer=False):
    
    # ---- (1) Load and prepeocessing ---- #

    PATH = args.dataPATH if args.dataPATH[-1] == '/' else args.dataPATH + '/'
    
    X_tr, y_tr = get_resized_data('train', PATH, args.resize)
    X_va, y_va = get_resized_data('val', PATH, args.resize)
    X_te, y_te = get_resized_data('test', PATH, args.resize)

    '''
    y_tr = one_hot_transformation(y_tr)
    y_va = one_hot_transformation(y_va)
    y_te = one_hot_transformation(y_te)
    '''

    X_tr, y_tr = torch.FloatTensor(X_tr), torch.LongTensor(y_tr)
    X_va, y_va = torch.FloatTensor(X_va), torch.LongTensor(y_va)
    X_te, y_te = torch.FloatTensor(X_te), torch.LongTensor(y_te)

    print('\nShapes of feature matrices (Train | Val | Test):', end='  ')
    print(X_tr.shape, X_va.shape, X_te.shape)
    print('Shapes of y label matrices (Train | Val | Test):', end='  ')
    print(y_tr.shape, y_va.shape, y_te.shape)

    try:
        os.makedirs(args.savePATH)
    except FileExistsError:
        pass
    
    # ---- (2) Training ---- #
    device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'
    d_trainers = {}
    trainer = Trainer(
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        device=device, model_name=args.model_name,
        model_hyper_param={
            'channels': args.channels,
            #'num_conv_layers': args.num_conv_layers,
            'filter_size': args.filter_size,
            'pooling_size': args.pooling_size,
            'hidden_sizes': args.hidden_sizes,
            'hidden_act': args.hidden_act,
            'output_size': one_hot_transformation(y_tr).shape[1]
        },
        optimizer=args.optimizer, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size,
        savePATH=args.savePATH, save_model=args.save_model, pretrained_model=args.pretrained_model,
        verbose=args.verbose, random_state=args.random_state)
    trainer.train(X_va, y_va, args.print_result_per_epochs, args.save_model)
    
    # ---- (3) Evaluating ---- #

    print('\nEvaluate the trained model on the testing set ...')
    print_accuracy(y_onehot_te, trainer.model.predict(X_te))
    
    if trainer.trained:
        trainer.plot_training('loss', args.plot_figsize)
        trainer.plot_training('accuracy', args.plot_figsize)
    print('')

    if args.save_trainer:
        fn = trainer.fn.replace('Accuracy', 'Trainer').replace('.txt', '.pkl')
        trainer.X_tr, trainer.y_tr = None, None   # Remove the training data from the trainer to avoid saving a heavy file. 
        d_trainers['trainer'] = trainer
        try:
            with open(fn, 'wb') as f:
                pickle.dump(d_trainers, f, pickle.HIGHEST_PROTOCOL)
            print('\nThe trainer is saved as', fn)
        except:
            print('\nThe trainer fails to be saved!!!!! (；ﾟДﾟ；)')

    if return_trainer:
        d_trainers['trainer'] = trainer
        return d_trainers

if __name__ == '__main__':
    args = init_arguments().parse_args()
    #np.cuda.Stream.null.synchronize()
    main(args)
