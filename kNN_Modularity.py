# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:55:35 2018

@author: icruicks
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community


def kNN_Modularity(distances):
    bestModularity =-np.infty
    bestNetwork = nx.Graph()
    bestPart =[]
    kBest =2
    np.fill_diagonal(distances, np.infty)
    
    for n in range(1, np.int(np.floor(np.log2(distances.shape[0])))):
        k = 2**n
        nodes = np.argpartition(distances, k)[:,:k]
        edges = []
        for i in range(nodes.shape[0]):
            for j in nodes[i,:]:
                edges.append((i,j))
                
        network = nx.Graph()
        network.add_nodes_from(nodes[:,0])
        network.add_edges_from(edges)
        
        currPart = community.best_partition(network)
        currModularity = community.modularity(currPart, network)
        if currModularity > bestModularity:
            bestModularity = currModularity
            bestNetwork = network
            bestPart = currPart
            kBest = k

    return np.array([bestPart[i] for i in range(nodes.shape[0])]), kBest, bestModularity, bestNetwork


affinity_test =np.random.rand(1000,1000)

groups, k , modularity, graph = kNN_Modularity(affinity_test)

print(groups)
print('\n')
print('Best Modularity: {}'.format(modularity))
nx.draw_networkx(graph)
