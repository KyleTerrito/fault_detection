""""
Parent class

Is the interfact with data. 

Provides methods for data preprocessing for faulDetection.

"""

#import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessing():
    def __init__(self):
        pass

    def load_data(self, path='data/processed/TEP0-2_labeled.xlsx'):
        #Load data
        data = pd.read_excel(path)
        print(data.head(10))

        labels = data.iloc[:, -1]

        data = data.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(data,
                                                            labels,
                                                            test_size=0.2)

        return X_train, X_test, y_train, y_test

        #scaling
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    def train_test_split(self, data, labels, test_size):

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size)

        return X_train, X_test, y_train, y_test
