import pandas as pd
import numpy as np
import os
import json

class RawTestData:
    def __init__(self, test_data_file_path="../../data/test/test.jsonl", test_annotated_pairs_file_path="../../data/test/final_test_pairs.csv"):
        self.test_data_file_path = test_data_file_path
        self.test_annotated_pairs_file_path = test_annotated_pairs_file_path
        self.test_data, self.test_annotated_pairs = self.load_data()

    def load_data(self):
        """
            Load test data and test annotated pairs from the given paths
            Returns:
                test_data: a dataframe from the jsonl file
                test_annotated_pairs: a dataframe from the task csv file
        """

        test_data = pd.read_json(self.test_data_file_path, lines=True, dtype={'pair_id': str})
        test_data = self._preprocess_test_data(test_data)

        test_annotated_pairs = pd.read_csv(self.test_annotated_pairs_file_path, dtype={'pair_id': str})

        return test_data, test_annotated_pairs

    @staticmethod
    def _preprocess_test_data(test_data):
        """
            Static method to preprocess test data:
                - extract id and text from n1_data and n2_data columns
                - drop duplicates
        """

        test_data['id1'] = test_data['n1_data'].apply(lambda x: x['id'])
        test_data['text1'] = test_data['n1_data'].apply(lambda x: x['text'])
        test_data['id2'] = test_data['n2_data'].apply(lambda x: x['id'])
        test_data['text2'] = test_data['n2_data'].apply(lambda x: x['text'])

        # drop n1_data and n2_data columns
        test_data = test_data.drop(columns=['n1_data', 'n2_data'])

        # drop duplicates
        test_data = test_data.drop_duplicates(subset='pair_id')

        return test_data

    def describe_data(self):
        """
            Describe test data and test annotated pairs
        """

        print("Test data (from the repo) shape: ", self.test_data.shape)
        print("Test annotated pairs (from the task) shape:", self.test_annotated_pairs.shape)
        print("*"*50)

        not_in_test_data = self.test_annotated_pairs[~self.test_annotated_pairs['pair_id'].isin(self.test_data['pair_id'])]
        not_in_test_annotated_pairs = self.test_data[~self.test_data['pair_id'].isin(self.test_annotated_pairs['pair_id'])]

        print("Number of pairs in test annotated data (task) but not in test data:", not_in_test_data.shape[0])
        print("Number of pairs in test data (repo) but not in test annotated data: ", not_in_test_annotated_pairs.shape[0])

        print("*"*50)

    def merge_data(self):
        """
            Merge test data and test annotated pairs on pair_id
            Returns:
                merged_test_data: a dataframe with columns: pair_id, id1, text1, id2, text2, Overall
        """

        merged_test_data = pd.merge(self.test_annotated_pairs, self.test_data, on="pair_id", how="inner")

        selected_columns = ["pair_id", "id1", "text1", "id2", "text2", "Overall"]
        merged_test_data = merged_test_data[selected_columns]

        print(f"Merged test data. Shape: {merged_test_data.shape}. Columns: {merged_test_data.columns}")
        #create a csv file with the merged_test_data
        merged_test_data.to_csv("../../data/test/merged_test_data.csv", index=False)    
        
        return merged_test_data


if __name__ == "__main__":
    raw_test_data = RawTestData()
    raw_test_data.describe_data()
    merged_test_data = raw_test_data.merge_data()