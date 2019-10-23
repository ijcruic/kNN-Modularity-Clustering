# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:33:33 2018

@author: icruicks
"""
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import kneighbors_graph
import numpy as np, networkx as nx
from sknetwork.clustering import Louvain, modularity

class kNN_network(BaseEstimator, ClusterMixin):

    def __init__(self, metric='euclidean', graph_type='symmetric'):
         self.best_modularity =-1
         self.best_network = nx.Graph()
         self.best_part =[]
         self.k_best =2
         self.metric=metric
         self.graph_type = graph_type
         
    def fit_predict(self, X):
    
        for n in range(1, np.int(np.floor(np.log2(X.shape[0])))):
            k = 2**n
            kNN = kneighbors_graph(X, metric=self.metric, mode='connectivity', n_neighbors=int(k))
            if self.graph_type == 'symmetric':
                kNN = kNN.minimum(kNN.T)
            elif self.graph_type=='assymetric':
                kNN = kNN.maximum(kNN.T)
            network = nx.from_scipy_sparse_matrix(kNN)
            clstr = Louvain()
            curr_part = clstr.fit(kNN).labels_
            random_kNN = nx.to_scipy_sparse_matrix(self._randomize_network(network, self.graph_type))
            curr_random_part = clstr.fit(random_kNN).labels_
            curr_modularity = modularity(kNN, curr_part) - modularity(random_kNN, curr_random_part)
            #curr_modularity = modularity(kNN, curr_part) - self._random_modularity(network)
            if curr_modularity > self.best_modularity:
                self.best_modularity = curr_modularity
                self.best_network = network
                self.best_part = curr_part
                self.best_k = k

        return self.best_part
    
    def _randomize_network(self, network, graph_type):
        n = network.number_of_nodes()
        p = nx.density(network)
        if graph_type =='directed':
            return nx.fast_gnp_random_graph(n, p, directed=True)
        else:
            return nx.fast_gnp_random_graph(n, p, directed=False)
    
    def _random_modularity(self, network):
        S = network.number_of_nodes()
        p = nx.density(network)
        return (1-2/np.sqrt(S))*(2/(p*S))**(2/3)
        
        
        
        