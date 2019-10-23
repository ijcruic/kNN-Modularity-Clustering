# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:42:17 2018

@author: icruicks
"""

import pandas as pd, networkx as nx, kNN_Modularity
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import adjusted_mutual_info_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

subGroupsDF = pd.DataFrame({'Nodes':y})
subGroupsDF.replace(to_replace={0:'Setosa', 1:'Versicolour',
                          2:'Virginica'}, inplace=True)

kNN = kNN_Modularity.kNN_network(metric='euclidean', graph_type='symmetric')
subgroups = kNN.fit_predict(X)
latent_network = kNN.best_network


print("The Adjusted Mutyal Information of k-NN Modularity Maximization on cluster label is: {}".format(adjusted_mutual_info_score(y, subgroups)))

labels = subGroupsDF['Nodes']
nx.draw(latent_network, node_color=subgroups, cmap=plt.cm.Set1, labels=labels)