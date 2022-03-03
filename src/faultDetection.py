"""
Child class of dataPreprocessing

Contains methods for Dimensionality reduction, clustering and fault detection

"""

import numpy as np
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import DBSCAN
from sklearn import metrics


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

    #TODO: add performTSNE, reconstructTSNE


class Clustering():
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

    def performDBSCAN(self, data, hyperparameters):

        cl_model = DBSCAN(eps=hyperparameters[0],
                          min_samples=hyperparameters[1])
        labels = cl_model.fit_predict(data)  #.labels_

        return labels

    #TODO: add performHDBSCAN, performSOMClustering


class FaultDetection():
    def __init__(self):
        pass

    #TODO: add fault detection methods and performance metrics