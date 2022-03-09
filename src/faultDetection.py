"""
Child class of dataPreprocessing

Contains methods for Dimensionality reduction, clustering and fault detection

"""

import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import hdbscan
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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

    def performGEN(self, method, data, hyperparameters):

        if 'PCA' in method:
            return self.performPCA(data, hyperparameters)
        elif 'UMAP' in method:
            return self.performUMAP(data, hyperparameters)

    def reconstructGEN(self, method, data, hyperparameters):

        if 'PCA' in method:
            return self.reconstructPCA(data, hyperparameters)
        elif 'UMAP' in method:
            return self.reconstructUMAP(data, hyperparameters)

    def performPCA(self, data, hyperparameters):

        n_components = hyperparameters[0]

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
                               min_dist=hyperparameters[1],
                               n_components=hyperparameters[2]).fit(data)

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

    def performGEN(self, method, data, hyperparameters):

        if 'KMEANS' in method:
            labels = self.performKMEANS(data, hyperparameters)

        elif 'HDBSCAN' in method:
            labels = self.performHDBSCAN(data, hyperparameters)

        elif 'DBSCAN' in method:
            labels = self.performDBSCAN(data, hyperparameters)

        return labels

    def performDBSCAN(self, data, hyperparameters):

        cl_model = DBSCAN(eps=hyperparameters[0],
                          min_samples=hyperparameters[1])

        labels = cl_model.fit_predict(data)  #.labels_

        return labels

    def performKMEANS(self, data, hyperparameters):

        cl_model = KMeans(n_clusters=hyperparameters[0], random_state=0)

        labels = cl_model.fit_predict(data)

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

        knn_model = KNeighborsClassifier(n_neighbors=hyperparameters)

        knn_model.fit(train_data, labels)

        return knn_model

    def predict(self, knn_model, test_data):

        predicted_labels = knn_model.predict(test_data)

        return predicted_labels

    def alignLabels(self,
                    true_labels,
                    predicted_labels,
                    majority_threshold_percentage=0.8,
                    print_reassignments=False):

        ground_truth_set = list(set(true_labels))
        predicted_labels_set = list(set(predicted_labels))
        aligned_labels_set = set(predicted_labels_set)
        aligned_labels = np.empty_like(predicted_labels)

        for label in range(len(predicted_labels_set)):
            #Takes one predicted cluster at a time
            this_label = predicted_labels_set[label]
            this_cluster_mask = predicted_labels == predicted_labels_set[label]
            this_pred_cluster = predicted_labels[this_cluster_mask]
            this_true_cluster = true_labels[this_cluster_mask]

            #Check to see if all members of this predicted cluster share the same ground truth label
            if not np.all(this_true_cluster == this_true_cluster[0]):
                unique, counts = np.unique(this_true_cluster,
                                           return_counts=True)
                majority_share = max(counts) / sum(counts)

                if majority_share > majority_threshold_percentage:
                    #If this cluster is split, but one class holds an 80% majority, let's assign them all to that class
                    majority_name = unique[np.where(counts == max(counts))][0]
                    aligned_labels[this_cluster_mask] = majority_name
                    if print_reassignments:
                        print(
                            f"Assigned cluster with {majority_share*100}% majority to cluster: {majority_name}"
                        )
                else:
                    #If the split is closer to 50-50, then let's separate this predicted cluster as a brand new cluster
                    new_cluster_name = max(aligned_labels_set) + 1
                    aligned_labels_set.add(new_cluster_name)
                    aligned_labels[this_cluster_mask] = new_cluster_name
                    if print_reassignments:
                        print(
                            f"Assigned predicted cluster:{this_label} to new cluster:{new_cluster_name}"
                        )
            else:
                #This cluster is consistent, so we can assign it to the cluster it matches
                aligned_labels[this_cluster_mask] = this_true_cluster[0]
                if print_reassignments:
                    print(
                        f"Assigned consistent cluster: {this_true_cluster[0]}")

        return aligned_labels

    def accuracy(self, true_labels, predicted_labels):

        confusion = metrics.multilabel_confusion_matrix(
            true_labels, predicted_labels)

        accuracy = metrics.accuracy_score(true_labels, predicted_labels)

        return confusion, accuracy
