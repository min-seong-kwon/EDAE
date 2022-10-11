# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""
import gc

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Subtract, Lambda, Reshape, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint, TensorBoard
import scipy
from scipy.stats import uniform
from keras import backend as K 

try:
    from keras.optimizers import RMSprop, Adam, SGD # old tf version
except:
    from tensorflow.keras.optimizers import RMSprop, Adam, SGD

import argparse
import numpy as np
import pandas as pd
from scipy.special import comb
import matplotlib.pyplot as plt
import sys
import os
from scipy.sparse import vstack, csc_matrix
# from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from baseline.DevNet.utils import dataLoading, aucPerformance
from sklearn.model_selection import train_test_split
from myutils import Utils
from myutils import find_closest
from myutils import takeClosest

import time


class AutoEncoder():
    def __init__(self, seed, model_name='AE', save_suffix='test', q=0.9, alpha=13., beta=2.):
        self.utils = Utils()
        self.device = self.utils.get_device()  # get device
        self.seed = seed
        self.MAX_INT = np.iinfo(np.int32).max

        # self.sess = tf.Session() #for old version tf
        self.sess = tf.compat.v1.Session()
        self.data_format = 0
        parser = argparse.ArgumentParser()
        parser.add_argument("--network_depth", choices=['2', '4'], default='2',
                            help="the depth of the network architecture")
        parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
        parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
        parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
        parser.add_argument("--runs", type=int, default=10,
                            help="how many times we repeat the experiments to obtain the average performance")
        parser.add_argument("--cont_rate", type=float, default=0.02,
                            help="the outlier contamination rate in the training data")
        parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
        parser.add_argument("--data_set", type=str, default='annthyroid', help="a list of data set names")
        parser.add_argument("--output", type=str,
                            default='./results/devnet_auc_performance_30outliers_0.02contrate_2depth_10runs.csv',
                            help="the output file path")
        parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
        # self.args = parser.parse_args()
        self.args, unknown = parser.parse_known_args()

        # network depth
        self.network_depth = int(self.args.network_depth)
        # random_seed = args.ramdn_seed
        self.threshold = 0
        self.alpha=alpha
        self.beta=beta
        self.q = q
        self.save_suffix = save_suffix
        print("alpha:",self.alpha)
        print("beta:",self.beta)
        print("q:",self.q)
        if not os.path.exists('baseline/ExpNet/model'):
            os.makedirs('baseline/ExpNet/model')
        self.ref = None # normal distribution reference, created for reusing across subsequent function calls

   
    def dev_network_s(self,input_shape, modelname):
        '''
        network architecture with one hidden layer
        '''
        x_input = Input(shape=input_shape)
        length = K.int_shape(x_input)[1]
       # print(length)
        
        en1 = Dense((length//2)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(x_input)
        en1 = Dropout(0.3)(en1)
        
        en2 = Dense((length//4)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(en1)
        en2 = Dropout(0.3)(en2)
        
        zz = Dense((length//8)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(en2) 
        zz = Dropout(0.1)(zz)
        
        de1 = Dense((length//4)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(zz)
        de1 = Dropout(0.3)(de1)
        
        de2 = Dense((length//2)+1, kernel_initializer='glorot_normal', activation='LeakyReLU')(de1)
        de2 = Dropout(0.3)(de2)
        
        de3 = Dense(length, kernel_initializer='glorot_normal',activation='LeakyReLU')(de2)
                
        sub_result = Subtract()([x_input, de3])
        cal_norm2 = Lambda(lambda x: tf.square(tf.norm(x,ord = 2,axis=1)))

        sub_norm2 = cal_norm2(sub_result)
        anomaly_score = Reshape((1,))(sub_norm2)
        
        model = Model(x_input,anomaly_score)
        model.load_weights(modelname)

        return model
        #1
#########################################################################################################################################################################################################        
    def dev_network_pretrain(self,input_shape):
        '''
        network architecture with one hidden layer
        '''
       
        x_input = Input(shape=input_shape)
        length = K.int_shape(x_input)[1]
        #print(length)
        
        en1 = Dense((length//2)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(x_input)
        en1 = Dropout(0.3)(en1)
        
        en2 = Dense((length//4)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(en1)
        en2 = Dropout(0.3)(en2)
        
        zz = Dense((length//8)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(en2) 
        zz = Dropout(0.1)(zz)
        
        de1 = Dense((length//4)+1, kernel_initializer='glorot_normal',activation='LeakyReLU')(zz)
        de1 = Dropout(0.3)(de1)
        
        de2 = Dense((length//2)+1, kernel_initializer='glorot_normal', activation='LeakyReLU')(de1)
        de2 = Dropout(0.3)(de2)
        
        de3 = Dense(length, kernel_initializer='glorot_normal',activation='LeakyReLU')(de2)
                
        sub_result = Subtract()([x_input, de3])
        cal_norm2 = Lambda(lambda x: tf.square(tf.norm(x,ord = 2,axis=1)))

        sub_norm2 = cal_norm2(sub_result)
        anomaly_score = Reshape((1,))(sub_norm2)
        
        model = Model(x_input,anomaly_score)
        
        return model

#########################################################################################################################################################################################################

    	
    def deviation_loss(self, y_true, y_pred):
        '''
        z-score-based deviation loss
        '''

        threshold=self.threshold 
        alpha=self.alpha
        beta=self.beta
        dev = y_pred
        inlier_loss = dev
        outlier_loss = alpha*K.exp(beta*(threshold-dev))

        return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)
#2
    
    def pretrain_loss(self, y_true, y_pred):
        '''
        z-score-based deviation loss
        '''

        inlier_loss = y_pred

        return K.mean((1 - y_true) * inlier_loss) 

        
#########################################################################################################################################################################################################
    def deviation_network(self, input_shape, network_depth,modelname):
        '''
        construct the deviation network-based detection model
        '''
        if network_depth == 4:
            model = self.dev_network_pretrain(input_shape)
            adm = Adam(lr=0.001)
            rms = RMSprop(clipnorm=1.,learning_rate=0.001)
            model.compile(loss=self.pretrain_loss, optimizer=rms)
        elif network_depth == 2:
            model = self.dev_network_s(input_shape,modelname)
            adm = Adam(lr=0.001)
            rms = RMSprop(clipnorm=1.,learning_rate=0.001)
            model.compile(loss=self.deviation_loss, optimizer=rms)
          #  model.summary()

        else:
            sys.exit("The network depth is not set properly")
        return model
        

#########################################################################################################################################################################################################

                             
    def auto_encoder_batch_generator_sup(self, x, inlier_indices, batch_size, nb_batch, rng):
        """auto encoder batch generator
        """
        self.utils.set_seed(self.seed)
        # rng = np.random.RandomState(rng.randint(self.MAX_INT, size = 1))
        rng = np.random.RandomState(np.random.randint(self.MAX_INT, size=1))
        counter = 0
        while 1:
            if self.data_format == 0:
                ref, training_labels = self.AE_input_batch_generation_sup(x, inlier_indices, batch_size, rng)
            else:
                ref, training_labels = self.input_batch_generation_sup_sparse(x, inlier_indices, batch_size, rng)
            counter += 1
            yield(ref, training_labels)
            if (counter > nb_batch):
                counter = 0

    def AE_input_batch_generation_sup(self, train_x, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        rng = np.random.RandomState(self.seed)

        dim = train_x.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = np.empty((batch_size, dim))
        n_inliers = len(inlier_indices)
        for i in range(batch_size):
            sid = rng.choice(n_inliers, 1)
            ref[i] = train_x[inlier_indices[sid]]
            training_labels[i] = train_x[inlier_indices[sid]]
        return np.array(ref), np.array(training_labels, dtype=float)

    def batch_generator_sup(self, x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
        """batch generator
        """
        rng = np.random.RandomState(rng.randint(self.MAX_INT, size = 1))
        counter = 0
        while 1:
            ref, training_labels = self.input_batch_generation_sup_3(x, outlier_indices, inlier_indices, batch_size, rng)
            counter += 1
            yield(ref, training_labels)
            if (counter > nb_batch):
                counter = 0


    def input_batch_generation_sup_3(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        #batchs of samples. This is for csv data.
        #Alternates between positive and negative pairs.
        dim = X_train.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = [0]*batch_size
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)

        r = uniform.rvs(size=int(batch_size/2))
        ad_score_samples = scipy.stats.gamma.ppf(
                    q = r, 
                    a = self.gamma_params[0], 
                    loc = self.gamma_params[1], scale = self.gamma_params[2])

        sid = []
        for score in ad_score_samples:
            sid.append(self.sorted_anomaly_score_indices[takeClosest(self.sorted_anomaly_score, score)])

        ref[0:batch_size:2] = X_train[inlier_indices[sid]]        
        training_labels[0:batch_size:2] = [0]*int(batch_size/2)

        sid = rng.choice(n_outliers, int(batch_size/2))
        ref[1:batch_size:2] = X_train[outlier_indices[sid]]
        training_labels[1:batch_size:2] = [1]*int(batch_size/2)
 
        return np.array(ref), np.array(training_labels, dtype=float)

    def input_batch_generation_sup_2(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        dim = X_train.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)

        for i in range(batch_size):
            if(i % 2 == 0):
                ###############################################################
                r = uniform.rvs(size=1)

                ad_score_sample = scipy.stats.gamma.ppf(
                    q = r, 
                    a = self.gamma_params[0], 
                    loc = self.gamma_params[1], scale = self.gamma_params[2])

                sid = find_closest(self.log_anomaly_scores, ad_score_sample)
                ###############################################################
                ref[i] = X_train[inlier_indices[sid]]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = X_train[outlier_indices[sid]]
                training_labels += [1]
        return np.array(ref), np.array(training_labels, dtype=float)

    def input_batch_generation_sup(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        dim = X_train.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(batch_size):
            if(i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = X_train[inlier_indices[sid]]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = X_train[outlier_indices[sid]]
                training_labels += [1]
        return np.array(ref), np.array(training_labels, dtype=float)

    def input_batch_generation_sup_sparse(self, X_train, outlier_indices, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for libsvm stored sparse data.
        Alternates between positive and negative pairs.
        '''
        ref = np.empty((batch_size))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(batch_size):
            if(i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = inlier_indices[sid]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = outlier_indices[sid]
                training_labels += [1]
        ref = X_train[ref, :].toarray()
        return ref, np.array(training_labels)

    def load_model_weight_predict(self, model_name, input_shape, network_depth, X_test):
        '''
        load the saved weights to make predictions
        '''
        model = self.deviation_network(input_shape, network_depth, model_name)
        model.load_weights(model_name)
        scoring_network = Model(inputs=model.input, outputs=model.output)

        scores = scoring_network.predict(X_test)
        np.set_printoptions(threshold=sys.maxsize)
      #  print(scores)
        
       # print(scores.shape)
        return scores


    def fit(self, X_train, y_train, pre_epoch=50,epoch=11, ratio=None):
        #index
        K.clear_session()
        gc.collect()

       
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]
        n_outliers = len(outlier_indices)
        print("Training size: %d, No. outliers: %d" % (X_train.shape[0], n_outliers))

        #set seed using myutils
        self.utils.set_seed(self.seed)
        #rng = np.random.RandomState(random_seed)
        rng = np.random.RandomState(self.seed)

        #start time
        self.input_shape = X_train.shape[1:]
        self.model = self.deviation_network(self.input_shape, 4, None)  # pretrain auto-encoder model
        print('autoencoder pre-training start....')
        
        self.model_name = os.path.join(os.getcwd(), 'baseline', 'ExpNet', 'model', 'pretrained_autoencoder_'+self.save_suffix+'.h5')
        # main/baseline/ExpNet/model/pretrained_autoencoder_test.h5
        
        checkpointer = ModelCheckpoint(self.model_name, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True)
        
        self.model.fit_generator(self.auto_encoder_batch_generator_sup(X_train, inlier_indices, self.args.batch_size, self.args.nb_batch, rng),
                                         steps_per_epoch=self.args.nb_batch, epochs=pre_epoch, callbacks=[checkpointer])
        
        
        return self
        
    def predict_score(self, X):
        score = self.load_model_weight_predict(self.model_name, self.input_shape, 2, X)
        # score = self.model.predict(X)
        
        return score
