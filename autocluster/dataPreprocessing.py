""""
Parent class

Is the interfact with data. 

Provides methods for data preprocessing for faulDetection.

"""

#import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split


class DataPreprocessing():
    def __init__(self):
        pass

    def load_data(self, path='data/processed/TEP0-2_labeled.xlsx', normalize_method=None):
        # Load data
        data = pd.read_excel(path)
        print(data.head(10))

        labels = data.iloc[:, -1]

        data = data.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(data,
                                                            labels,
                                                            test_size=0.2,
                                                            # stratify=labels,
                                                            )
        print(set(y_test))


        X_train_normal, X_test_normal = self.scale(X_train, X_test, normalize_method)

        return X_train_normal, X_test_normal, y_train, y_test

    # scaling
    def scale(self, X_train, X_test, normalize_method):
        if normalize_method == None:
            return(X_train, X_test)

        elif normalize_method == "Z":
            scaler = StandardScaler().fit(X_train)
            X_train_normal = scaler.transform(X_train)
            X_test_normal = scaler.transform(X_test)
        
        elif normalize_method == "mean":
            scaler = StandardScaler(with_std=False).fit(X_train)
            X_train_normal = scaler.transform(X_train)
            X_test_normal = scaler.transform(X_test)

        return (X_train_normal, X_test_normal)
    '''
    removed user define function as is does the same as sklearn's import
    '''
    # def train_test_split(self, data, labels, test_size):

    #     X_train, X_test, y_train, y_test = train_test_split(
    #         data, labels, test_size=test_size)

    #     return X_train, X_test, y_train, y_test
