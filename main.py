'''
Deep Learning - HW3: LeNet
Jay Liao (re6094028@gs.ncku.edu.tw)
'''
 
import time, os
import cupy as np
import numpy as np
import pandas as pd
from source.args import init_arguments
from source.utils import *
from source.trainer import *

def main(args, return_trainer=False):
    
    # ---- (1) Load and prepeocessing ---- #

    PATH = args.dataPATH if args.dataPATH[-1] == '/' else args.dataPATH + '/'
    
    X_tr, y_tr = get_resized_data('train', PATH, args.resize)
    X_va, y_va = get_resized_data('val', PATH, args.resize)
    X_te, y_te = get_resized_data('test', PATH, args.resize)
    
    X_tr = resize_channel(X_tr)
    X_va = resize_channel(X_va)
    X_te = resize_channel(X_te)
    
    y_onehot_tr = one_hot_transformation(y_tr)
    y_onehot_va = one_hot_transformation(y_va)
    y_onehot_te = one_hot_transformation(y_te)

    print('\nShapes of feature matrices (Train | Val | Test):', end='  ')
    print(X_tr.shape, X_va.shape, X_te.shape)
    print('Shapes of y label matrices (Train | Val | Test):', end='  ')
    print(y_onehot_tr.shape, y_onehot_va.shape, y_onehot_te.shape)

    try:
        os.makedirs(args.savePATH)
    except FileExistsError:
        pass
    
    # ---- (2) Training ---- #

    d_trainers = {}
    trainer = Trainer(
        X_tr=X_tr, y_tr=y_onehot_tr,
        X_te=X_te, y_te=y_onehot_te,
        model_name=args.model_name,
        model_hyper_param={
            'input_dim': args.input_dim,
            'num_conv_layers': args.num_conv_layers,
            'conv_params': {'filter_num': args.nums_filter, 'filter_size': args.sizes_filter, 'pad': args.pads, 'stride': args.strides},
            'pooling_size': args.pooling_size,
            'pooling_stride': args.pooling_stride,
            'hidden_sizes': args.hidden_sizes,
            'hidden_act': args.hidden_act,
            'output_size': y_onehot_tr.shape[1]
        },
        optimizer=args.optimizer,
        optimizer_param={'lr': args.lr},
        epochs=args.epochs,
        batch_size=args.batch_size,
        savePATH=args.savePATH,
        save_model=args.save_model,
        pretrained_model=args.pretrained_model,
        verbose=args.verbose,
        random_state=args.random_state)
    trainer.train(X_va, y_onehot_va, args.print_result_per_epochs, args.save_model)
    
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
    np.cuda.Stream.null.synchronize()
    main(args)