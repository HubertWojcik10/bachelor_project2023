import os
from data_preprocessing.train_data import RawTrainData
import pandas as pd

class Baseline:
    def __init__(self, params_dict: dict):
        self.train_data = pd.read_csv(params_dict["train_data_path"])
        self.test_data = pd.read_csv(params_dict["test_data_path"])
        self.rounding_strategy = params_dict["rounding_strategy"]

    def run(self):
        print(self.train_data.head())
