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

    def load_data(self):
        #Load data
        data = pd.read_excel("data", header=None).dropna()
        print(data.head(10))

        data = data.to_numpy()

        return data

    #scaling
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    def train_test_split(self, data, labels, test_size):

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

        return X_train, X_test, y_train, y_test

