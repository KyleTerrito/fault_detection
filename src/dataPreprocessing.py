""""
Parent class

Is the interfact with data. 

Provides methods for data preprocessing for faulDetection.

"""
#import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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