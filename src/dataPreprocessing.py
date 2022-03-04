""""
Parent class

Is the interfact with data. 

Provides methods for data preprocessing for faulDetection.

"""
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessing():
    def __init__(self):
        pass

    def train_test_split(self, data, labels, test_size):

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

        return X_train, X_test, y_train, y_test