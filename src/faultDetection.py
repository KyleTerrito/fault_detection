"""
Child class of dataPreprocessing

Contains methods for Dimensionality reduction, clustering and fault detection

"""

import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.manifold import TSNE
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class DR():
    '''
    Contains methods for dimensionality reduction and data reconstruction

    -All DR methods must receive (data, hyperparameters) as inputs and return (model, reduced data)
    
    -All reconstruction methods must receive (model, dr_data) and return (reconstructed data)
    '''
    def __init__(self):
        pass

    def performPCA(self, data, hyperparameters):

        n_components = hyperparameters

        pca_model = PCA(n_components)
        pca_model.fit(data)
        dr_data = pca_model.transform(data)

        return pca_model, dr_data

    def reconstructPCA(self, pca_model, dr_data):

        rc_data = pca_model.inverse_transform(dr_data)

        return rc_data

    def performUMAP(self, data, hyperparameters):

        umap_model = umap.UMAP(n_neighbors=hyperparameters[0],
                               init="random",
                               n_components=hyperparameters[1],
                               min_dist=hyperparameters[2]).fit(data)

        dr_data = umap_model.transform(data)

        return umap_model, dr_data

    def reconstructUMAP(self, umap_model, dr_data):

        rc_data = umap_model.inverse_transform(dr_data)

        return rc_data


class Clustering():
    #TODO: performSOMClustering
    '''
    Contains methods for clustering and clustering performance metrics

    -Metrics should be small is better

    -All clustering methods must receive (data, hyperparameters) as inputs and return (labels)
    
    -All reconstruction methods must receive (data, labels) and return (metric score)
    '''
    def __init__(self):
        pass

    def silmetric(self, data, labels):
        return metrics.silhouette_score(data, labels)

    def mstscore(self, data, labels):

        if labels is None:
            labels = [0] * len(data[:, 0])

        data = np.column_stack((data, np.array(labels)))

        scores = []

        for label in labels:
            cluster_list = []
            for i in range(len(data[:, 0])):
                if data[i, -1] == label:
                    cluster_list.append(data[i])

            cluster = np.asarray(cluster_list)

            dist_matrix = squareform(pdist(cluster))

            X = csr_matrix(dist_matrix)
            Tcsr = minimum_spanning_tree(X)

            # print(Tcsr)

            average_dist = np.mean(Tcsr)
            n_edges = np.count_nonzero(Tcsr.toarray())

            # print(average_dist)
            # print(n_edges)

            score = n_edges / average_dist

            scores.append(score)

        total_score = np.sum(np.asarray(scores))

        #print(total_score)

        return total_score

    def performDBSCAN(self, data, hyperparameters):

        cl_model = DBSCAN(eps=hyperparameters[0],
                          min_samples=hyperparameters[1])
        labels = cl_model.fit_predict(data)  #.labels_

        return labels

    def performHDBSCAN(self, data, hyperparameters):

        cl_model = hdbscan.HDBSCAN(
            min_cluster_size=hyperparameters[0],
            min_samples=hyperparameters[1],
            cluster_selection_epsilon=hyperparameters[2])

        labels = cl_model.fit_predict(data)

        return labels


class FaultDetection():
    def __init__(self):
        pass

    #TODO: add fault detection methods and performance metrics