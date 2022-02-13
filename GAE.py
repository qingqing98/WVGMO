# -*- coding: utf-8 -*-
"""

"""

from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE, KLD
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda,BatchNormalization
from spektral.layers import  GCNConv
from keras.optimizers import Adam

from tensorflow.keras.initializers import GlorotUniform
from layers import *
import numpy as np
sess = tf.compat.v1.Session()

class GAE(tf.keras.Model):

    def __init__(self, X, adj, adj_n, hidden_dim=128, latent_dim=100, dec_dim=None, adj_dim=32):
        super(GAE, self).__init__()
        if dec_dim is None:
            dec_dim = [128,256,512]
        self.latent_dim = latent_dim
        self.X = X
        self.adj = np.float32(adj)
        self.adj_n = np.float32(adj_n)
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.sparse = False
        self.batch_size = 16
        
        initializer = GlorotUniform(seed=7)
        
        #encoder
        X_input = Input(shape=self.in_dim)
        h = Dropout(0.2)(X_input)
        A_in = Input(shape=self.n_sample)
        h = GCNConv(channels=hidden_dim,  kernel_initializer=initializer, activation="relu")([h, A_in])
        z_mean = GCNConv(channels=latent_dim, kernel_initializer=initializer)([h, A_in])
        
        self.encoder = Model(inputs=[X_input, A_in], outputs=z_mean, name="encoder")
        
        # Adjacency matrix decoder
        
        dec_in = Input(shape=latent_dim)
        h = Dense(units=adj_dim, activation=None)(dec_in)
        h = Bilinear()(h)
        dec_out = Lambda(lambda z: tf.nn.sigmoid(z))(h)
        self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoder1")
        
         # Expression matrix decoder

        decx_in = Input(shape=latent_dim)
        h = Dense(units=dec_dim[0], activation="relu")(decx_in)
        h = Dense(units=dec_dim[1], activation="relu")(h)
        h = Dense(units=dec_dim[2], activation="relu")(h)
        decx_out = Dense(units=self.in_dim)(h)
        self.decoderX = Model(inputs=decx_in, outputs=decx_out, name="decoderX")
        
    #Training model
    def alt_train(self, epochs=100,lr=5e-4,W_a=0.5, W_x=1, info_step=8, n_update=8, centers=None):
        
        def get_GAE_loss(X, newX, beta):#beta=10
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X)* B, 2))  
        
        # Training
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        for epoch in range(0, epochs):
            
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder([self.X, self.adj_n])
                X_out = self.decoderX(z)
                A_out = self.decoderA(z)
                A_rec_loss1 = get_GAE_loss(self.adj_n, A_out, 5)
                A_rec_loss2 = tf.reduce_mean(MSE(self.adj_n, A_out))
                A_rec_loss=A_rec_loss1+A_rec_loss2
                X_rec_loss = tf.reduce_mean(MSE(self.X, X_out))
                tot_loss = W_a * A_rec_loss + W_x * X_rec_loss

            vars = self.trainable_weights
            grads = tape.gradient(tot_loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

            if epoch % info_step == 0:
                print("Epoch", epoch, " X_rec_loss: ", X_rec_loss.numpy(), " A_rec_loss: ", A_rec_loss.numpy())
    
    def embedding(self,count,adj_n):
        if self.sparse:
            adj_n=tfp.math.dense_to_sparse(adj_n)
        return np.array(self.encoder([count,adj_n]))
        
    def rec_A(self, count, adj_n):
        h = self.encoder([count, adj_n])
        rec_A = self.decoderA(h)
        return np.array(rec_A)
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        