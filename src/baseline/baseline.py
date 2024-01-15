import os
from data_preprocessing.train_data import RawTrainData
import pandas as pd

class Baseline:
    def __init__(self):
        self.train_data = pd.read_csv("../data/train/train.csv")
        self.test_data = pd.read_csv("../data/test/final_test_pairs.csv")

    def run(self):
        print(self.train_data.head())
