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

    def __init__(self, metric='euclidean', graph_type='symmetric', 
                 clustering_alg='louvain', clustering_iterations=5):
         self.best_modularity =-1
         self.best_network = nx.Graph()
         self.best_part =[]
         self.k_best =2
         self.metric=metric
         self.graph_type = graph_type
         self.clustering_alg = clustering_alg
         self.clustering_iterations = clustering_iterations
         
    def fit_predict(self, X):
    
        for n in range(1, np.int(np.floor(np.log2(X.shape[0])))):
            k = 2**n
            kNN = kneighbors_graph(X, metric=self.metric, mode='connectivity', n_neighbors=int(k))
            if self.graph_type == 'symmetric':
                kNN = kNN.minimum(kNN.T)
            elif self.graph_type=='assymetric':
                kNN = kNN.maximum(kNN.T)
            network = nx.from_scipy_sparse_matrix(kNN)
            
            if self.clustering_alg =='leiden':                
                curr_part, avg_modularity = self._get_leiden_partition_and_modularity(kNN)
                random_kNN = nx.to_scipy_sparse_matrix(self._randomize_network(network, self.graph_type))
                curr_random_part, avg_random_modularity = self._get_leiden_partition_and_modularity(random_kNN)
                curr_modularity = avg_modularity - avg_random_modularity
                
            elif self.clustering_alg =='louvain':
                curr_part, avg_modularity = self._get_louvain_partition_and_modularity(kNN)
                random_kNN = nx.to_scipy_sparse_matrix(self._randomize_network(network, self.graph_type))
                curr_random_part, avg_random_modularity = self._get_louvain_partition_and_modularity(random_kNN)
                curr_modularity = avg_modularity - avg_random_modularity
            else:
               raise Exception('select valid clustering algorithm. Options are leiden and louvain')
                
            #curr_modularity = avg_modualrity - self._random_modularity(network)
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
    
    def _get_louvain_partition_and_modularity(self, kNN):
        cluster_iterations = [Louvain()]*self.clustering_iterations
        parts = [clstr.fit(kNN).labels_ for clstr in cluster_iterations]
        modularities = [modularity(kNN, part) for part in parts]
        best_part = parts[np.argmax(modularities)]
        avg_modularity = np.mean(modularities)
        return best_part, avg_modularity
        
    def _get_leiden_partition_and_modularity(self, kNN):
        import leidenalg as la
        graphs =[self._scipy_to_igraph(kNN)]*self.clustering_iterations
        leiden_results = [la.find_partition(G, la.CPMVertexPartition) for G in graphs]
        parts =[]
        for result in leiden_results:
            temp_parts = np.zeros(kNN.shape[1])
            for i in range(len(result)):
                for j in range(len(result[i])):
                    temp_parts[result[i][j]] = i
            parts.append(temp_parts)
        modularities = [modularity(kNN, part) for part in parts]
        best_part = parts[np.argmax(modularities)]
        avg_modularity = np.mean(modularities)
        return best_part, avg_modularity
    
    def _scipy_to_igraph(self, matrix):
        from igraph import Graph
        if self.graph_type =='directed':
            directed=True
        else:
            directed = False
        sources, targets = matrix.nonzero()
        weights = matrix[sources, targets]
        return Graph(list(zip(sources, targets)), directed=directed, edge_attrs={'weight': weights})
        
        
        