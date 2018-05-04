# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:33:33 2018

@author: icruicks
"""

import pandas as pd, numpy as np, networkx as nx, os
import community
from matplotlib import pyplot

def kNetwork(frequencies):
    bestModularity =-np.infty
    bestNetwork = nx.Graph()
    bestPart =[]
    kBest =2
    np.fill_diagonal(frequencies, -np.infty)
    
    for n in range(1, np.int(np.floor(np.log2(frequencies.shape[0])))):
        k = 2**n
        nodes = np.argpartition(frequencies, -3)[:,-3:]
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


frequencyMatrix = pd.read_csv(os.path.join("Data","SCM_Frequency_Network.csv"))
labels = frequencyMatrix.iloc[:,0]
subGroupsDF = pd.DataFrame({'Nodes':labels})
networkDF = pd.DataFrame(index=labels)
subGroupsDF['sub groups'], k, modularity, network = kNetwork(frequencyMatrix.iloc[:,1:].as_matrix())

nx.draw(network)
nx.draw_networkx_labels(network, pos=nx.spring_layout(network), labels=labels.to_dict())

networkDF = nx.to_pandas_adjacency(nx.relabel_nodes(network, labels.to_dict()))
networkDF.to_csv(os.path.join("Data","SCM_latent_network.csv"))
subGroupsDF.to_csv(os.path.join("Data","SCM_louvain_groups.csv"))