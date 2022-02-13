# -*- coding: utf-8 -*-
"""

"""
import argparse
import sys
import numpy as np
import time
import os


import tensorflow as tf
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense,Lambda, Layer,BatchNormalization,Dropout, Activation
from keras.models import Model
from keras.losses import mse
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from os.path import splitext, basename, isfile



class WVAE():
    def __init__(self, X_shape, n_components, epochs=100):
        self.epochs = epochs
        self.batch_size = 16
        sample_size = X_shape[0]
        self.epochs = 20
        self.n_components = n_components
        self.shape = X_shape[1]

    def train(self, X):
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), seed=0)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        encoding_dim = self.n_components
        original_dim = X.shape[1]
        input = Input(shape=(original_dim,))
        encoded = Dense(encoding_dim)(input)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)
        z_mean = Dense(encoding_dim)(encoded)
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        z_log_var = Dense(encoding_dim)(encoded)
        z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
        decoded = Dense(encoding_dim, activation='relu')(z)
        output = Dense(original_dim, activation='sigmoid')(decoded)
        wvae = Model(input, output)
        encoder = Model(input, z)
        reconstruction_loss = mse(input, output)
        reconstruction_loss *= original_dim
        wd_loss=K.square(tf.norm(z_mean))+tf.linalg.trace(K.exp(z_log_var)+tf.eye(batch,dim)+2*tf.sqrt(K.exp(z_log_var)))
        wd_loss=K.sqrt(wd_loss)
        wd_loss *= -0.5
        wvae_loss = K.mean(reconstruction_loss + wd_loss)
        wvae.add_loss(wvae_loss)
        wvae.compile(optimizer=Adam())
        print(len(wvae.layers))
        print(wvae.count_params())
        wvae.fit(X, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return encoder.predict(X)
    
class WVAE_API(object):
    def __init__(self, model_path='./model/', epochs=200, weight=0.001):
        self.model_path = model_path
        self.score_path = './score/'
        self.epochs = epochs
        self.batch_size = 16
        self.weight = weight
    
 # feature extract
    def feature_wvae(self, df_ori, n_components=100, b_decomposition=True):
        if b_decomposition:
            X = self.encoder_wvae(df_ori, n_components)
            print(X)
            fea = pd.DataFrame(data=X, index=df_ori.index,
                               columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = df_ori.copy()
        print("feature extract finished!")
        return fea
    
    def encoder_wvae(self, df, n_components=100):
        vae = WVAE(df.shape, n_components, self.epochs)
        return vae.train(df)
    
    def kmeans(self, n_clusters=28):
        model = KMeans(n_clusters=n_clusters, random_state=0)
        return model
    
tf.compat.v1.disable_v2_behavior()
def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='WVAE')
    parser.add_argument("-i", dest='file_input', default="/input/UCEC.list",
                        help="file input")
    parser.add_argument("-e", dest='epochs', type=int, default=200, help="Number of iterations")
    parser.add_argument("-m", dest='run_mode', default="wvae", help="run_mode: feature, cluster")
    parser.add_argument("-n", dest='cluster_num', type=int, default=-1, help="cluster number")
    parser.add_argument("-w", dest='disc_weight', type=float, default=1e-4, help="weight")
    parser.add_argument("-t", dest='type', default="UCEC", help="cancer type: KIRC, UVM")
    args = parser.parse_args()
    model_path = '/result/' + args.type + '.h5'
    WVAE = WVAE_API(model_path, epochs=args.epochs, weight=args.disc_weight)
    cancer_dict = {'BLCA': 5, 'KIRC':5,
                   'LUAD': 5,'SKCM': 5,'UCEC': 3, 'UVM': 5}
    
    if args.run_mode == 'wvae':
        cancer_type = args.type
        if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
        dfs = []
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_tmp_file = '/fea/' + cancer_type + '/' + base_file + '.fea'
            dfs.append(pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=','))
        X = pd.concat(dfs, axis=0).T
        print(X.head(5))
        print(X.shape)
        fea_save_file = '/fea/UCEC.wvae'
        start_time = time.time()
        vec = WVAE.feature_wvae(X, n_components=100)
        vec.to_csv(fea_save_file, header=True, index=True, sep='\t')
        
        df = pd.DataFrame(data=[time.time() - start_time])
        out_file = '/result/UCEC.wvae.time'
        df.to_csv(out_file, header=True, index=False, sep=',')
        if isfile(fea_save_file):
            X = pd.read_csv(fea_save_file, header=0, index_col=0, sep='\t')
            X['wvae'] = WVAE.kmeans(args.cluster_num).fit_predict(X.values) + 1
            X = X.loc[:, ['wvae']]
            out_file = '/result/' + cancer_type + '.wvae'
            X.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print('file does not exist!')
    
    
    
if __name__ == "__main__":
    main()   
    
    
    

    
    
    
    
    