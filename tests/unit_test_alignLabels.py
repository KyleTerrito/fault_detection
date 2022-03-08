import numpy as np
import pandas as pd
from sklearn import preprocessing

from src.faultDetection import FaultDetection
from src.faultDetection import Clustering
from src.dataPreprocessing import DataPreprocessing


def test():

    path = 'data/processed/TEP0-2_labeled.xlsx'

    data = np.asarray(pd.read_excel(path, sheet_name='Sheet1', index_col=0))
    
    data_values = data[:,:-1]
    data_labels = data[:,-1]
    le = preprocessing.LabelEncoder().fit(data_labels)
    data_labels = le.transform(data_labels)

    DP = DataPreprocessing()
    train_data, test_data, train_labels, test_labels = DP.train_test_split(data_values, data_labels, test_size=0.20)


    CL = Clustering()
    hyperparameters = [15, 1, 0.5]

    cluster_labels = CL.performHDBSCAN(train_data, hyperparameters)

    FD = FaultDetection()
    hyperparameters = 10

    #knn_model = FD.trainKNN(train_data, train_labels, hyperparameters)

    #FD.alignLabels(test_labels, cluster_labels, 10)
    


    hyperparameters = 10

    FD = FaultDetection()
    knn_model = FD.trainKNN(train_data, cluster_labels, hyperparameters)
    pred_labels = FD.predict(knn_model, test_data)

    aligned_prediction_labels = FD.alignLabels(test_labels, pred_labels, majority_threshold_percentage=0.8, print_reassignments=True)

    accuracy, confusion = FD.accuracy(test_labels, aligned_prediction_labels)

    print("Testing Accuracy:\n", accuracy)
    print(confusion)


    return None