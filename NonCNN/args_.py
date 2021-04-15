import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='DL hw1: Image classification')
    parser.add_argument('--seed', type=int, default=4028)
    parser.add_argument('--dataPATH', type=str, default='./data/', help='The path where the data (feature matrices or original images) are be put.')
    parser.add_argument('--feature_types', nargs='+', default=['Histogram', 'SIFT', 'SURF'], help='Which types of methods are going to be used to extrach features?')
    parser.add_argument('--n_Ranges', type=int, default=5, help='How many bars does the the global color histogram have?')
    parser.add_argument('--models', nargs='+', default=['OneLayerPerceptron', 'TwoLayerPerceptron', 'TwoLayerNet'], help='Which kinds of models are going to be trained?')
    parser.add_argument('--n_jobs', type=int, default=-1, help='How may workers are going to be used for training XGB classifier?')
    parser.add_argument('--save_trainer', action='store_true', default=True, help='Whether to save the trainer trainer. Only available for an NN trainer.')
    parser.add_argument('--save_models', action='store_true', default=True, help='Whether to save the trained models')
    parser.add_argument('--savePATH', type=str, default='./output/', help='The path to store the outputs, including models, plots, and training and evalution results.')
    parser.add_argument('--hidden_size', type=int, default=32, help='Dimension of the hidden state of the TwoLayerNet')
    parser.add_argument('--hidden_layer_act', type=str, default='ReLU', choices=['None', 'ReLU', 'Sigmoid', 'tanh'])
    parser.add_argument('--output_layer_act', type=str, default='SoftmaxWithCrossEntropyLoss', choices=['ReLU', 'Sigmoid', 'SoftmaxWithCrossEntropyLoss'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Momentum', 'AdaGrad', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=120, help='No. of epochs')
    parser.add_argument('--print_result_per_epochs', type=int, default=30, help='How many epochs that the evaluated results be printed once?')
    parser.add_argument('--pretrained_model', type=str, default='', help='File name of the pretained model that is going to keep being trained or to be evaluated. Set an empty string if not using a pretrained model.')
    parser.add_argument('--plot_figsize', nargs='+', type=int, default=[8,6])
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--resize', nargs='+', type=int, default=(64, 64), help='The target size of images')

    return parser