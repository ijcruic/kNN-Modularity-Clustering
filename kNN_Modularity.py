# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:33:33 2018

@author: icruicks
"""
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np, networkx as nx
import community

class kNetwork(BaseEstimator, ClusterMixin):

    def __init__(self):
         self.bestModularity =-1
         self.bestNetwork = nx.Graph()
         self.bestPart =[]
         self.kBest =2
         
    def fit_predict(self, distances):
        distances = np.copy(distances)
        np.fill_diagonal(distances, np.infty)
    
        for n in range(1, np.int(np.floor(np.log2(distances.shape[0])))):
            k = 2**n
            nodes = np.argpartition(distances, k)[:,:k]
            edges = []
            for i in range(nodes.shape[0]):
                for j in nodes[i,:]:
                    edges.append((i,j))
                    
            network = nx.DiGraph()
            network.add_nodes_from(nodes[:,0])
            network.add_edges_from(edges)
            network = network.to_undirected(reciprocal=False)
            
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

        return np.array([self.bestPart[i] for i in range(nodes.shape[0])]), self.kBest, self.bestModularity, self.bestNetwork
    
    def _randomize_network(self, network):
        adj_matrix = nx.to_numpy_matrix(network)
        np.transpose(np.random.shuffle(np.transpose(adj_matrix)))
        return nx.from_numpy_matrix(adj_matrix)
    
    def _random_modularity(self, network):
        S = network.number_of_nodes()
        p = nx.density(network)
        return (1-2/np.sqrt(S))*(2/(p*S))**(2/3)
        
        
        
        