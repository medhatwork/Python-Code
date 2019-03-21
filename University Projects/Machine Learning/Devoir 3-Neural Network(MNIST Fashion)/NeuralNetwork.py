
# Presente par Ahmed Mohamed et Sofiene Fehri
import numpy as np
from maths_functions import *
from tqdm import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold 
from skimage.feature import hog
from sklearn.utils import shuffle
import time


class NeuralNetwork:
    def __init__(self, sizes):
        np.random.seed(0)

        self.sizes      = sizes
        self.model      = {}
        self.b          = np.array([np.zeros((x, 1))  for x in sizes[1:]])
        self.d          = self.sizes[0]  
        self.s          = self.sizes[-1]      
        self.w          = np.array([(np.random.randn(x, y) / np.sqrt(x)) for x,y in zip(self.sizes[1:],self.sizes[:-1])])
        for w in self.w:
            w = np.random.uniform(-1/np.sqrt(w.shape[1]), 1/np.sqrt(w.shape[1]),(w.shape[0],w.shape[1]))

    def fprop(self,x):
        model = {}
        for j in range(1,len(self.sizes)):
            weight_j = 'W{}'.format(j)  
            bias_j = 'b{}'.format(j)
            model[weight_j] = self.w[j-1]
            model[bias_j] = self.b[j-1]

        model['a0'] = x
        for i in range(1,len(self.sizes)):
           
            z = model['W'+str(i)] @ model['a'+str(i-1)] + model['b'+str(i)]
            model['z'+str(i)] = z
            a = g_x(self.h_act,z) if i < (len(self.sizes)-1) else g_x(self.out_act,z) 
            model['a'+str(i)] = a
        
       
        self.model = model
       

    def bprop(self,y):
        delta3 = self.model['a'+str(len(self.sizes)-1)] - y
        self.model['delta'+str(len(self.sizes))] = delta3
                
        for j in range(len(self.sizes)-1,0,-1):
            a = self.model['a'+str(j-1)]
            z = self.model['z'+str(1)]
            delta = self.model['delta'+str(j+1)]
            self.model['grad_w'+str(j)] = delta @ a.T 
            #_lambda *    
            self.model['grad_b'+str(j)] = np.sum(delta, axis=1, keepdims=True)
            if (j>1):
                self.model['delta'+str(j)] = ((delta.T@self.model['W'+str(j)]).T * (gPrime_x(self.h_act,z))) 
                + (self.L1_lambda * np.divide(np.array([w for w in self.w]),np.abs(np.array([w for w in self.w]))))
                + (self.L2_lambda*np.array([w for w in self.w]))
    
    def train(self, data, epochs, minibatch_size, L1_lambda, L2_lambda, lrate, h_act, out_act, test, images,
              check_grad, vector_minibatch, valid_percent):
        
        self.minibatch_size    = minibatch_size
        self.L1_lambda         = L1_lambda
        self.L2_lambda         = L2_lambda
        self.lrate             = lrate
        self.h_act             = h_act
        self.out_act           = out_act 
        self.train_errors      = []
        self.check_grad        = check_grad
        self.epochs            = epochs
        self.test              = test
        self.validation_errors = []
        self.test_errors       = []
        self.valid_percent     = valid_percent
            
            
        self.train_inputs = data['train_inputs']
        self.train_labels = data['train_labels']
        self.test_inputs = data['test_inputs']
        self.test_labels = data['test_labels']

        self.onehot = self.onehot_encode(self.train_labels)
        self.test_onehot = self.onehot_encode(self.test_labels)

        
        if images :
            self.train_inputs = np.array(self.train_inputs)
            sqrt = int(np.sqrt(self.train_inputs.shape[1]))
            length = self.train_inputs.shape[0]
            self.train_inputs = self.train_inputs.reshape((length,sqrt,sqrt))
            self.train_inputs = self.images_preprocessing(self.train_inputs)
            

            self.test_inputs = np.array(self.test_inputs)
            length = self.test_inputs.shape[0]
            self.test_inputs = self.test_inputs.reshape((length,sqrt,sqrt))
            self.test_inputs = self.images_preprocessing(self.test_inputs)


            if self.sizes[0] != self.train_inputs.shape[1]:
                self.sizes[0] = self.train_inputs.shape[1]
                self.d          = self.sizes[0]  
                self.b          = np.array([np.zeros((x, 1))  for x in self.sizes[1:]])
                self.w          = np.array([(np.random.randn(x, y) / np.sqrt(x)) for x,y in zip(self.sizes[1:],self.sizes[:-1])])
                for w in self.w:
                    w = np.random.uniform(-1/np.sqrt(w.shape[1]), 1/np.sqrt(w.shape[1]),(w.shape[0],w.shape[1]))

        if valid_percent > 0:
            self.cross_validation_and_test()

        else : 
            self.training_type()

    def cross_validation_and_test(self):
        folds = int(np.floor(1/self.valid_percent))
        kf = KFold(n_splits=folds) 
        kf.get_n_splits(self.train_inputs)
        i = 0 
        train_inputs = self.train_inputs
        train_labels = self.train_labels

        for train_index, test_index in kf.split(np.array(self.train_inputs)):
            self.train_inputs = train_inputs
            self.train_labels = train_labels
            train_index = train_index[train_index<len(self.train_inputs)]
            test_index = test_index[test_index<len(self.train_inputs)]
            try:
                self.train_inputs, self.train_labels = self.train_inputs[train_index], self.train_labels[train_index]
                self.validation_inputs, self.validation_labels = self.train_inputs[test_index], self.train_labels[test_index]
                self.validation_onehot = self.onehot_encode(self.validation_labels)
                self.training_type()
                tqdm.write("\nFold {0}/{1} complete : \nCross validation error : {2}\n".format(i+1,folds,validation_error))
                
            except:
                continue
            i+=1

    def training_type(self):

        if self.minibatch_size == 0:
            self.train_SGD()

        elif self.minibatch_size < len(self.train_inputs) and vector_minibatch:
            self.train_vector_minibatch()

        elif self.minibatch_size < len(self.train_inputs) and not vector_minibatch:
            self.train_minibatch()

        else : self.train_batch()


    def train_SGD(self):


        x = self.train_inputs
        y = self.onehot

        if(type(self.train_errors) != 'list'):
            self.train_errors = list(self.train_errors)

        tic = time.time()
        for j in tqdm(range(self.epochs)):
    
            self.train_inputs,self.train_labels = shuffle(self.train_inputs, self.train_labels, random_state=0)
            total_loss_per_epoch = 0
            stopped = False
            data_left = 0
            L = 0

            for i in range(len(x)):
                x_i = x[i].reshape(self.d,1)
                y_i = y[i].reshape((self.s,1))
                self.fprop(x_i)
                self.bprop(y_i)
                self.update_param()
                L += self.loss_function(y_i)
            
            total_loss_per_epoch += L

            self.train_errors.append(total_loss_per_epoch)

            tqdm.write("Epoch {0} complete : \nLoss : {1}\nMean Training Loss : {2}\n".format(j,total_loss_per_epoch,
                (total_loss_per_epoch/len(x))))

            if self.valid_percent > 0 :
                evaluation = self.evaluate(self.validation_inputs, self.validation_onehot)
                tqdm.write("Cross validation score: {0}\n".format(evaluation))
                self.validation_errors.append(1-evaluation)


            if self.test:
                evaluation = self.evaluate(self.test_inputs, self.test_onehot)
                tqdm.write("Test score: {0}\n".format(evaluation))
                self.test_errors.append(1-evaluation)
            
            
        toc = time.time()

        if self.check_grad :
            self.verif_gradient(x,y)
        

    def train_minibatch(self,validation):

        x = self.train_inputs
        y = self.onehot

        total_loss_per_epoch = 0
        stopped = False
        data_left = 0
        if(type(self.train_errors) != 'list'):
            self.train_errors = list(self.train_errors)

        tic = time.time()

        for j in tqdm(range(self.epochs)):

            L = 0
            self.train_inputs,self.train_labels = shuffle(self.train_inputs, self.train_labels, random_state=0)

            for i in range(0, self.train_inputs.shape[0] -data_left, self.minibatch_size):
                x_minibatch = self.train_inputs[i:i+ self.minibatch_size]
                y_minibatch = self.onehot[i:i+ self.minibatch_size]
                data_left = len(self.train_inputs)%self.minibatch_size

                if data_left == 0:
                    stopped = True

                for i in range(len(x)):
                    x_i = x[i].reshape(self.d,1)
                    y_i = y[i].reshape((self.s,1))
                    self.fprop(x_i)
                    self.bprop(y_i)
                    self.update_param()
                    L += self.loss_function(y_i)

            
            total_loss_per_epoch = L
            self.train_errors.append(total_loss_per_epoch)

            tqdm.write("Epoch {0} complete : \nLoss : {1}\nMean Training Loss : {2}".format(j,total_loss_per_epoch,
                (total_loss_per_epoch/len(x))))

            if self.test:
                test_errors, evaluation = self.evaluate(self.test_inputs, self.test_onehot)
                tqdm.write("Test score: {0}  ".format(evaluation))
                self.test_errors.append(1-evaluation)
        
        toc = time.time()

        print("\nTime taken for non vectorial mini batch :", str(toc - tic))

        if self.check_grad :
            self.verif_gradient(x,y)


    # def train_vector_miniBatch(self):

       
    #     for j in tqdm(range(self.epochs)):
    
    #         x = self.train_inputs
    #         y = self.onehot

            
    #         total_loss_per_epoch = 0
    #         stopped = False
    #         data_left = 0
    #         L = 0

    #         idx_train = int(self.train_inputs.shape[0] % self.minibatch_size)
    #         idx_test = int(self.test_inputs.shape[0] % self.minibatch_size)
    #         train_inputs = self.train_inputs[:-idx_train]
    #         onehot = self.onehot[:-idx_train]
    #         test_inputs = self.test_inputs[:-idx_test]
    #         test_onehot = self.test_onehot[:-idx_test]

    #         for i in range(0, train_inputs.shape[0] - data_left, self.minibatch_size):
    #                 x_minibatch = train_inputs[i:i+ self.minibatch_size]
    #                 new_shape = self.sizes[0]
    #                 if x_minibatch.shape[0] < self.minibatch_size:
    #                     continue
    #                 y_minibatch = onehot[i:i+self.minibatch_size]
    #                 data_left = len(train_inputs)-self.minibatch_size
    #                 x_minibatch = x_minibatch.reshape((new_shape),1)
    #                 y_minibatch = y_minibatch.reshape((new_shape),1)
    #                 self.fprop(x_minibatch)
    #                 self.bprop(y_minibatch)
    #                 self.update_param()
    #                 L += self.loss_function(y_minibatch)

    #         total_loss_per_epoch += L
    #         self.train_errors.append(total_loss_per_epoch)

    #         if self.check_grad :
    #             self.verif_gradient(x,y)

    #         if self.test :
    #             for i in range(0, test_inputs.shape[0] - data_left, new_shape):
    #                 test_inputs[i] = test_inputs[i:i+new_shape]
    #                 test_onehot[i] = test_onehot[i:i+new_shape]
    #                 test_inputs[i] = test_inputs.reshape((new_shape),1)
    #                 test_onehot[i] = test_onehot.reshape((new_shape/2),2)
    #                 data_left = len(train_inputs)-self.minibatch_size


    #             test_errors, evaluation = self.evaluate(self.test_inputs, self.test_onehot)
    #             tqdm.write("Test score: {0}  ".format(evaluation))
    #             self.test_errors.append(test_errors)

    #         tqdm.write("Epoch {0} complete : \nLoss {1}\n ".format(j,total_loss_per_epoch))


    # def train_batch(self):

      
    def onehot_encode(self,y):
        y = y.reshape(len(y), 1)
        onehot_encoder = OneHotEncoder(categories='auto',sparse=False)
        return onehot_encoder.fit_transform(y)
    
    def verif_gradient(self,x,y,epsilon= 10**-5):
        for i in range(len(x)):
            x_i = x[i].reshape((self.d,1))
            y_i = y[i].reshape((self.s,1))
            self.fprop(x_i)
            L = self.loss_function(y_i)
            self.bprop(y_i)
            gradients_w = [self.model['grad_w'+str(i+1)] for i in range(len(self.w))]
            gradients_b = [self.model['grad_b'+str(i+1)] for i in range(len(self.w))]
            ratio_w = []
            ratio_b = []
            for k,matrix_weight in enumerate (self.w):
                shape = matrix_weight.shape
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        matrix_weight[i][j] -= epsilon                    
                        self.fprop(x_i)
                        L2 = self.loss_function(y_i)
                        estimated_gradient = (L-L2)/epsilon
                        matrix_weight[i][j] += epsilon
                        ratio_w.append(np.float64((estimated_gradient+epsilon)/(gradients_w[k][i][j]+epsilon)))   

       
        print(ratio_w)

    def update_param(self):
        for i in range(len(self.w)-1,-1,-1):
             self.w[i] -= self.lrate * self.model['grad_w'+str(i+1)]
             self.b[i] -= self.lrate * self.model['grad_b'+str(i+1)]

    def loss_function(self,y):
        p = self.model['a'+str(len(self.sizes)-1)]
        x = np.sum(p*y, axis = 0)
        return (((-np.log(x)) + 0.5 *self.L1_lambda * np.abs(self.w)) + 0.5*self.L2_lambda*(self.w**2))[0][0][0]

    def evaluate(self, test_data, test_labels, func_type='sgd'):
        activations = []
        test_results = []

        for i in range(len(test_data)):
            if func_type == 'sgd':
                self.fprop(np.array([test_data[i]]).T)
                test_results.append(((np.argmax(np.sum(self.model['a'+str(len(self.sizes)-1)],axis=1)))
                                     == np.argmax(test_labels[i])))

            else :
                self.fprop(np.array(test_data.T))
                test_results.append(((np.argmax(np.sum(self.model['a'+str(len(self.sizes)-1)],axis=1)))
                                     == np.argmax(test_labels)))
            
        return (np.sum(np.array(test_results)*1))/len(test_labels)

    def predict(self, test_inputs):
        test_results = []
        for i in range(len(test_inputs)):
            self.fprop(np.array([test_inputs[i]]).T)
            test_results.append(np.max((np.sum(self.model['a'+str(len(self.sizes)-1)],axis=1))))
        test_results = np.array(test_results)

        return test_results.reshape((len(test_results),1))

    def images_preprocessing(self,images):
        list_hog_fd = []
        for feature in tqdm(images):
                fd = hog(feature, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
                list_hog_fd.append(fd)
        return np.array(list_hog_fd, 'float64')

    def figures_plot(self,plot_type,title):
        if 'error' in plot_type:
            self.plot_errors = True
            self.plot_errors_title = title

        elif 'decision' in plot_type:
            self.plot_decision = True
            self.plot_decision_title = title

    