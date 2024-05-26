import torch.nn.functional as F
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from utils.clusteringPerformance import similarity_function, StatisticClustering

"""
Some other functions
"""

def spectral_clustering(points, k, gnd, repnum=10):
    W = similarity_function(points)
    Dn = np.diag(1 / np.power(np.sum(W, axis=1), -0.5))
    L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
    eigvals, eigvecs = LA.eig(L)
    eigvecs = eigvecs.astype(float)
    indices = np.argsort(eigvals)[:k]
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])

    [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(k_smallest_eigenvectors, gnd, k)
    print("ACC, NMI, ARI, Purity, Fscore, Precision, Recall: ", ACC, NMI, ARI, Purity, Fscore, Precision, Recall)
    return [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]

def GaussianNoise(x, stddev=0.3):
    gauss = np.random.normal(.0, stddev)
    return x * gauss

def GaussianNoise1(x, stddev=0.1):
    gauss = np.random.normal(.0, stddev)
    return x + gauss

def clustering(z, gnd, k):
    [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(z, gnd, k)
    print("ACC, NMI, Purity, ARI, Fscore, Precision, Recall : ", ACC, NMI, Purity, ARI, Fscore, Precision, Recall)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)