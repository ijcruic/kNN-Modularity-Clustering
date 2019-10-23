# k-NN Network Modularity Maximization Clustering

## Overview
k-NN Network Modularity Maximization is a clustering technique proposed initially by Ruan that iteratively constructs k-NN graphs, finds sub groups in those graphs by modularity maxmimization, and finally selects the graph and sub groups that have the best modularity. k-NN graph construction is done from an affinity matrix (which is a matrix of pairwise similarity between each entity and every other entity) where each entity receives a connection to their k most similar neighbors neighbors.

## Usage
The current implementaton is built as an Sci-Kit learn clustering module. So the code is similarly called as an sklearn clustering technique. The input to the method is the data matrix itself. Hyperparameters include what metric whould be used to measure nearness of the different entities (options are anything from scipy.distance module), and what type of latent graph should be found; options are 'direct' for a directed kNN, 'symmetric' for a mutual or symmetric kNN (edge exists between two entities only if each entity is in the k nearest neighbors of the other entity), and 'assymetric' for an assymetric kNN (edge exists between two entities if either entity is in the k nearest neighbiors of the other entity).

```python
import kNN_Modularity
kNN = kNN_Modularity.kNN_network
subgroups = kNN.fit_predict(X)
latent_network = kNN.best_network
```

Once key difference bewteen the original formulatio by Ruan and this implementation is that I am using Louvain modularity maximization for finding the sub groups, as it is a much faster routine than those used in the original paper (i.e. Qcut or HQcut). Also, the null-model modularity for absolute modularity computation can be found either by doing Louvain on a randomly rewired version of the newtork, or by the analytic formula for the modularity of an Erdos Renyi network of the same size and density.

The output will be the subgroup assignments for each entity in a numpy array. One can also access the latent network, in a networkx graph' by the 'best_network' attribute.

There is an example usage on the Iris datset also included in the code files.

## Future Work
The algorithm greedily and globally determines k, which can result in some connections forming that do not neccesarily make sense. Additionally, maximizing modularity alone with possibly disconnected graphs will result in a greater number of sub groups than would be expected. Modularity will be higher for graph with more connected components than fewer, given the same density.

The algorithm could benefit from a relaxed determination of k, whereby k could be different for each entity. Also, multiobjective optimization would also help with modularity and disconnected graphs. 

## References
* Ruan, J.: A fully automated method for discovering community structures in high
dimensional data. In: 2009 Ninth IEEE International Conference on Data Mining.
pp. 968{973 (Dec 2009). https://doi.org/10.1109/ICDM.2009.141

* Guimera R, Sales-Pardo M, Amaral LN. Modularity from fluctuations in random graphs and
complex networks. Physical Review E. 2004; 70:025101.

* Campedelli GM, Cruickshank I, Carley KM (2019) Detecting Latent Terrorist Communities Testing a Gower’sSimilarity-Based Clustering Algorithm for Multi-partite Networks. In: Aiello LM, Cherifi C, Cherifi H, LambiotteR, Li ́o P, Rocha LM (eds) Complex Networks and Their Applications VII, vol 812, Springer InternationalPublishing, Cham, pp 292–303.

