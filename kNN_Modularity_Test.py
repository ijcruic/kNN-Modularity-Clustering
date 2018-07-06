# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:42:17 2018

@author: icruicks
"""

import pandas as pd, networkx as nx, kNN_Modularity
from matplotlib import pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import cdist

iris = datasets.load_iris()
X = iris.data
y = iris.target

distanceMatrix = cdist(X, X)
subGroupsDF = pd.DataFrame({'Nodes':y})
subGroupsDF.replace(to_replace={0:'Setosa', 1:'Versicolour',
                          2:'Virginica'}, inplace=True)

kNetwork = kNN_Modularity.kNetwork()
subGroupsDF['sub groups'], k, modularity, network = kNetwork.fit_predict(distanceMatrix)

subGroupsDF = subGroupsDF.reindex(network.nodes)
labels = subGroupsDF['Nodes']
networkDF = pd.DataFrame(index=labels)
nx.draw(network, node_color=pd.Categorical(labels).codes, cmap=plt.cm.Set1)
nx.draw_networkx_labels(network, pos=nx.spring_layout(network), 
                        labels=subGroupsDF['sub groups'].to_dict())

networkDF = nx.to_pandas_adjacency()