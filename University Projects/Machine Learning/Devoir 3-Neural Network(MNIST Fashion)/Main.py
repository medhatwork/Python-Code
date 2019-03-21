# Presente par Ahmed Mohamed et Sofiene Fehri

import numpy as np
import argparse
from NeuralNetwork import *
from utils import mnist_reader
import models_plot


parser = argparse.ArgumentParser()
parser.add_argument('-test_percent', required = True, type = float, help="Test percentage")
parser.add_argument('-valid_percent', type = float, help="Validation Percentage")
parser.add_argument('-image', type=bool, help = "Specify the type as image")
parser.add_argument('-mnist', type=bool, help = "Specify that we're treating mnist images")
parser.add_argument('-csv', type=bool, help = "Specify the type as csv")
parser.add_argument('-path', type=str, required = True, help="Path of data")
parser.add_argument('-sizes', type=int, required = True, nargs='+', help="Number of layers and number of nurones per layer as list")
parser.add_argument('-epochs', type=int, required = True, help="Number of epochs")
parser.add_argument('-lrate', type=float, required = True, help="learning rate")
parser.add_argument('-minibatch_size', type=int, help="minibatch_size default = 0 || if 0 then we use all the SGD and if \
                     equal to dataset size we use b")
parser.add_argument('-L1_lambda', type=float, help="Lambda for L1 regularization default = 0 \
                      L1 and L2 gives elastic net regularization")
parser.add_argument('-L2_lambda', type=float, help="Lambda for L2 regularization default = 0")
parser.add_argument('-h_act', type=str, help="Hidden layer activation default relu")
parser.add_argument('-out_act', type=str, help="Output layer activation default softmax")
parser.add_argument('-test', type=str, help="Test default False")
parser.add_argument('-check_grad', type=bool, help="Check gradient")
parser.add_argument('-vector_minibatch', type=bool, help="Check gradient")
parser.add_argument('-test_plot_boundaries', type=bool, help="Plot decision boudaries for multiple hyperparamters")

args = parser.parse_args()

def main():
    #load data
    valid_percent = 0
    image = False
    mnist = True
    csv = False
    lambda1 = 0
    lambda2 = 0
    h_act = 'relu'
    out_act = 'softmax'
    test = False
    minibatch_size = 0
    images = True
    check_grad = False
    vector_minibatch = False
    test_plot_boundaries = False

    path = args.path
    sizes = args.sizes
    epochs = args.epochs
    lrate  = args.lrate
    test_percent = args.test_percent/100
    valid_percent = 0

    if args.valid_percent : valid_percent = args.valid_percent/100
    if args.image         : image = args.image
    if args.mnist         : mnist = args.mnist
    if args.csv           : csv = args.csv
    if args.L1_lambda     : lambda1 = args.L1_lambda
    if args.L2_lambda     : lambda2 = args.L2_lambda
    if args.minibatch_size: minibatch_size = args.minibatch_size
    if args.h_act     : h_act = args.h_act
    if args.out_act     : out_act = args.out_act
    if args.test     : test = args.test
    if args.check_grad     : check_grad = args.check_grad    
    if args.vector_minibatch    : vector_minibatch = args.vector_minibatch
    if args.test_plot_boundaries    : test_plot_boundaries = args.test_plot_boundaries


    if csv or 'txt' in path or '.csv' in path :
        images = False
        data = load_csv(path,test_percent,valid_percent)

    elif mnist == False:
        data = load_images(path,test_percent,valid_percent,mnist = False) 
    else :
        data = load_images(path,test_percent,valid_percent) 

    if not test_plot_boundaries:

        model = NeuralNetwork(sizes)

        if len(sizes) < 3 :
            str2 = 'a model with no hidden layers'
        elif len(sizes) == 3:
            str2 = 'a model with number of neurones = '+str(sizes[1])+' in one hidden layer'
        else:
            str2 = 'a model with number of neurones = '
            str2 += str([(str(x)+', ') for x in sizes[1:-2]])
            str2 += str(size[-1])+' nurones in '+str(len(sizes)-2)+' hidden layers'

        model.train(data, epochs , minibatch_size, lambda1, lambda2, lrate, h_act, out_act , test, 
                    images,check_grad, vector_minibatch,valid_percent)
        model.figures_plot('errors',('Errors for NN of '+str2))
        model.figures_plot('decision',('Decision boundaries for NN of '+ str2))
        models_plot.plot([model])

    else :
        premature_stop = [1,5,10,20,50]
        weight_decay = [0.0005,.001,0.01,0.1,1]
        hidden_sizes = [2,3,6,12]
        networks = list([NeuralNetwork([2,x,2]) for x in hidden_sizes])
        nets = []

        for i,net in enumerate(networks):
            net.train(data, epochs , minibatch_size, lambda1, lambda2, lrate, h_act, out_act , test, 
                      images,check_grad, vector_minibatch,valid_percent)
            net.figures_plot('errors','Errors for NN of '+str(hidden_sizes[i])+' neurones in hidden layer')
            net.figures_plot('decision',('Decision boundaries for NN of '+str(hidden_sizes[i])+' neurones in hidden layer'))
            nets.append(net)
            
        for p_stop in premature_stop:
            nnetwork = NeuralNetwork(sizes)
            nnetwork.train(data, p_stop , minibatch_size, lambda1, lambda2, lrate, h_act, out_act , test, 
                           images,check_grad, vector_minibatch, valid_percent)
            nnetwork.figures_plot('errors','Errors for NN with premature stop after '+str(p_stop)+' epochs')
            nnetwork.figures_plot('decision',('Decision boundaries for NN with premature stop after '+str(p_stop)+' epochs'))
            nets.append(nnetwork)

        for w in weight_decay:
            nnetwork = NeuralNetwork(sizes)
            nnetwork.train(data, epochs , minibatch_size, lambda1, w, lrate, h_act, out_act , 
                           test, images,check_grad, vector_minibatch, valid_percent)
            nnetwork.figures_plot('errors','Errors for NN with '+str(w)+'  as weight decay(Lambda for L2)')
            nnetwork.figures_plot('decision',('Decision boundaries for NN with '+str(w)+'  as weight decay(Lambda for L2)'))
            nets.append(nnetwork)
        models_plot.plot(nets)

def load_images(path,test_percent,valid_percent,mnist=True):
    datasets = {}
    if mnist:
        x_train, y_train = mnist_reader.load_mnist(path, kind='train')
        x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
        train_labels = y_train.reshape((y_train.shape[0],1))
        test_labels = y_test.reshape((y_test.shape[0],1))

        train_inputs = x_train/255
        test_inputs  = x_test/255

        datasets = {'train_inputs':train_inputs,'train_labels':train_labels,
                'test_inputs':test_inputs,'test_labels':test_labels}

    
    return datasets


def load_csv(path,test_percent,valid_percent):
    data = np.loadtxt(path)
    train_inputs = data[:,:-1]
    train_labels = data[:,-1]
    train_inputs,test_inputs = create_datasets(train_inputs,test_percent)
    train_labels,test_labels = create_datasets(train_labels,test_percent)
    datasets = {'train_inputs':train_inputs,'train_labels':train_labels,
                'test_inputs':test_inputs,'test_labels':test_labels}
    
    return datasets

def create_datasets(data, percent):
    idx = 1-int(percent * len(data))
    return data[:idx], data[idx:]


if __name__ == '__main__':
    main()