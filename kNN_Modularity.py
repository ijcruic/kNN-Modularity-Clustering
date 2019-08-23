# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:33:33 2018

@author: icruicks
"""
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import kneighbors_graph
import numpy as np, networkx as nx
import community

class kNN_network(BaseEstimator, ClusterMixin):

    def __init__(self, metric='euclidean', symmetrize=True):
         self.bestModularity =-1
         self.bestNetwork = nx.Graph()
         self.bestPart =[]
         self.kBest =2
         self.metric=metric
         self.symmetrize=symmetrize
         
    def fit_predict(self, X):
    
        for n in range(1, np.int(np.floor(np.log2(X.shape[0])))):
            k = 2**n
            kNN = kneighbors_graph(X, metric=self.metric, mode='connectivity', n_neighbors=int(k))
            if self.symmetrize:
                kNN = kNN.minimum(kNN.T)
            else:
                kNN = kNN.maximum(kNN.T)
            network = nx.from_scipy_sparse_matrix(kNN)
            currPart = community.best_partition(network)
            randomNetwork = self._randomize_network(network)
            currRandomPart = community.best_partition(randomNetwork)
            currModularity = community.modularity(currPart, network) - community.modularity(currRandomPart, randomNetwork)
            #currModularity = community.modularity(currPart, network) - self._random_modularity(network)
            if currModularity > self.bestModularity:
                self.bestModularity = currModularity
                self.bestNetwork = network
                self.bestPart = currPart
                self.kBest = k

        return np.array(list(self.bestPart.values()))
    
    def _randomize_network(self, network):
        adj_matrix = nx.to_numpy_matrix(network)
        np.transpose(np.random.shuffle(np.transpose(adj_matrix)))
        return nx.from_numpy_matrix(adj_matrix)
    
    def _random_modularity(self, network):
        S = network.number_of_nodes()
        p = nx.density(network)
        return (1-2/np.sqrt(S))*(2/(p*S))**(2/3)
        
        
        
        