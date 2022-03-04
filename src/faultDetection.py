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
from sklearn import KNeighborsClassifier
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

        #Divide the dataset into clusters
        cluster_list = []
        for i in range(len(labels)):
            cluster_i = []
            for j in range(len(data[:, 0])):
                if data[j, -1] == labels[i]:
                    cluster_i.append(data[j])
            cluster_list.append(cluster_i)

        #clusters = np.asarray(cluster_list)

        for cluster in cluster_list:

            dist_matrix = squareform(pdist(cluster))

            X = csr_matrix(dist_matrix)
            Tcsr = minimum_spanning_tree(X)

            # print(Tcsr)

            n_edges = np.count_nonzero(Tcsr.toarray())
            average_dist = np.sum(Tcsr) / n_edges

            #print(average_dist)
            # print(n_edges)

            score = average_dist  # / n_edges
            #print(f'cluster {label}, score = {score}')

            scores.append(score)

        total_score = np.sum(np.asarray(scores)) / len(labels)
        #print(total_score)

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
    '''
    Contains methods for classification and fault detection performance metrics

    -Metrics should be small is better

    -All classification methods must receive (data, labels, hyperparameters) as inputs
                                 and return (classification model)
    
    -All prediction methods must receive (classification model, test data) and return (predicted labels)
    '''
    def __init__(self):
        pass

    def trainKNN(self, train_data, labels, hyperparameters):

        knn_model = KNeighborsClassifier(n_neighbors=hyperparameters[0])
        
        knn_model.fit(train_data, labels)

        return knn_model

    def predict(self, knn_model, test_data):

        predicted_labels = knn_model.predict(test_data)

        return predicted_labels

    def accuracy(self, true_labels, predicted_labels):

        confusion = metrics.multilabel_confusion_matrix(true_labels, predicted_labels)

        accuracy = metrics. accuracy_score(true_labels, predicted_labels)

        return confusion, accuracy




