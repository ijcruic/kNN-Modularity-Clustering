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
         self.bestModularity =-np.infty
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
            currModularity = community.modularity(currPart, network)
            if currModularity > self.bestModularity:
                self.bestModularity = currModularity
                self.bestNetwork = network
                self.bestPart = currPart
                self.kBest = k

        return np.array([self.bestPart[i] for i in range(nodes.shape[0])]), self.kBest, self.bestModularity, self.bestNetwork