# -*- coding: utf-8 -*-
"""

"""
"Here is an example of gae integration of the latent characteristic information of each group learned by wvae, and then identifying cancer subtypes."


import os
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import norm
import scanpy as sc
from GAE import *
from preprocessing import *
from sklearn.cluster import KMeans
from sklearn import metrics

data1=pd.read_csv("..../fea//cancer_CN.VAE",delim_whitespace=True);
data2=pd.read_csv("..../fea//cancer_meth.VAE",delim_whitespace=True);
data3=pd.read_csv("..../fea//cancer_miRNA.VAE",delim_whitespace=True);
data4=pd.read_csv("..../fea//cancer_rna.VAE",delim_whitespace=True);

count=np.hstack((data1,data2,data3,data4))
adj,adj_n=get_adj(count)

model=GAE(count,adj=adj,adj_n=adj_n)
model.alt_train(epochs=100)
Y=model.embedding(count,adj_n)

n_clusters = 5

labels = KMeans(n_clusters=n_clusters, n_init=100).fit_predict(Y)

sli=metrics.silhouette_score(Y,labels)
ch=metrics.calinski_harabasz_score(Y, labels)
db=metrics.davies_bouldin_score(Y,labels)



