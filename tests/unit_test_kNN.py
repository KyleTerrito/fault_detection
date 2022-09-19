import numpy as np
import pandas as pd
from autocluster.faultDetection import FaultDetection as FD
from autocluster.dataPreprocessing import DataPreprocessing as DP


def test():

    path = 'data/processed/TEP0-2_labeled.xlsx'

    data = np.asarray(pd.read_excel(path, sheet_name='Sheet1', index_col=0))

    data_values = data[:,:-1]
    data_labels = data[:,-1]

    train_data, test_data, train_labels, test_labels = DP().train_test_split(data_values, data_labels, test_size=0.2)

    hyperparameters = [10]

    knn = FD()
    model = knn.trainKNN(train_data, train_labels, hyperparameters)
    pred_labels = knn.predict(model, test_data)
    accuracy, confusion = knn.accuracy(test_labels, pred_labels)

    print("Testing Accuracy:\n", accuracy)
    print(confusion)


    return None