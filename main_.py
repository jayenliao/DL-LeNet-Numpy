'''
Deep Learning - HW2 for HW3
Jay Liao (re6094028@gs.ncku.edu.tw)
'''
 
import time, os
import numpy as np
import pandas as pd
from NonCNN.args_ import init_arguments
from NonCNN.utils import *
from NonCNN.feature_extraction import *
from NonCNN.trainers import * 

def main_(args, feature_type, return_trainer=False):
    PATH = args.dataPATH if args.dataPATH[-1] == '/' else args.dataPATH + '/'
    fn = PATH + 'X_' + str(args.resize[0]) + '_' + feature_type + '_'
    if feature_type == 'Histogram':
        fn += str(args.n_Ranges) + '_'
    print('======= Feature type:', feature_type, '=======\n')
    try:
        print('Loading extracted features matrices ...', end=' ')
        X_tr, y_tr = np.load(fn + 'tr.npy', allow_pickle=True), load_original_data('train', PATH)[1]
        X_va, y_va = np.load(fn + 'va.npy', allow_pickle=True), load_original_data('val', PATH)[1]
        X_te, y_te = np.load(fn + 'te.npy', allow_pickle=True), load_original_data('test', PATH)[1]
        print('Done!')
    except:
        print('Fail! QAQ')
        #print('Loading the original images ...')
        img_list_tr, y_tr = get_resized_data('train', PATH, args.resize)
        img_list_va, y_va = get_resized_data('val', PATH, args.resize)
        img_list_te, y_te = get_resized_data('test', PATH, args.resize)
        
        print('\nExtracting features ...')
        t0 = time.time()
        X_tr = get_features_for_images(resize_channel(img_list_tr), feature_type, args.n_Ranges)
        X_va = get_features_for_images(resize_channel(img_list_va), feature_type, args.n_Ranges)
        X_te = get_features_for_images(resize_channel(img_list_te), feature_type, args.n_Ranges)
        tdiff = time.time() - t0
        print('Time cost of feature extraction: %8.2fs' % tdiff)
        
        np.save(fn + 'tr.npy', X_tr)
        np.save(fn + 'va.npy', X_va)
        np.save(fn + 'te.npy', X_te)

    print('\nShapes of feature matrices (Train | Val | Test):', end='  ')
    print(X_tr.shape, X_va.shape, X_te.shape)
    y_onehot_te = one_hot_transformation(y_te)

    d_trainers = {}
    for model in args.models:
        try:
            os.makedirs(args.savePATH)
        except FileExistsError:
            pass

        if model == 'TwoLayerPerceptron':
            args.hidden_layer_act = 'None'

        if model in ['rf', 'xgb']:
            args.save_trainer = False
            trainer = NonNNtrainer(X_tr=X_tr, y_tr=y_tr, feature_type=feature_type, model_=model, n_jobs=args.n_jobs, random_state=args.seed)
            trainer.train(save_model=args.save_models, savePATH=args.savePATH, pretrained_model=args.pretrained_model)
            print('\nEvaluate the trained model on the testing set ...')
            print_accuracy(y_onehot_te, trainer.model.predict_proba(X_te))

        else:
            y_onehot_tr, y_onehot_va = one_hot_transformation(y_tr), one_hot_transformation(y_va)
            print('Shapes of y label matrices (Train | Val | Test):', end='  ')
            print(y_onehot_tr.shape, y_onehot_va.shape, y_onehot_te.shape)
            
            trainer = NNtrainer(X_tr, y_onehot_tr, X_te, y_onehot_te, feature_type, model, args.hidden_size, args.hidden_layer_act, args.output_layer_act, args.optimizer, args.lr, args.seed)
            trainer.train(
                batch_size=args.batch_size,
                epochs=args.epochs,
                save_trainer=args.save_trainer,
                save_model=args.save_models,
                savePATH=args.savePATH,
                X_va=X_va,
                y_va=y_onehot_va,
                print_result_per_epochs=args.print_result_per_epochs,
                pretrained_model=args.pretrained_model,
                verbose=args.verbose
            )
            
            print('\nEvaluate the trained model on the testing set ...')
            print_accuracy(y_onehot_te, trainer.model.predict(X_te))

            if trainer.trained:
                if model != 'NaivePerceptron':
                    trainer.plot_training('loss', args.plot_figsize)
                trainer.plot_training('accuracy', args.plot_figsize)
        print()

        if args.save_trainer:
            fn = trainer.fn.replace('Accuracy', 'Trainer').replace('.txt', '.pkl')
            trainer.X_tr, trainer.y_tr = None, None   # Remove the training data from the trainer to avoid saving a heavy file. 
            try:
                with open(fn, 'wb') as f:
                    pickle.dump(trainer, f, pickle.HIGHEST_PROTOCOL)
                print('\nThe trainer is saved as', fn)
            except:
                print('\nThe trainer fails to be saved!!!!! (；ﾟДﾟ；)')

        if return_trainer:
            d_trainers[model] = trainer

    if return_trainer:
        return d_trainers

if __name__ == '__main__':
    arr = np.array([1, 2, 3])
    print(type(arr))
    args_ = init_arguments().parse_args()
    for feature_type in args_.feature_types:
        main_(args_, feature_type)