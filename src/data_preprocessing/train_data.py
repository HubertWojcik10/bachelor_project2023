import json
import csv 
import pandas as pd


class RawTrainData:
    def __init__(self, train_data_file_path="../../data/train/train.jsonl"):
         self.train_data_file_path = train_data_file_path
         self.reader_csv(self.train_data_file_path, output_path="../../data/train/train.csv")   

    # Open the JSON file for reading
    def reader_csv(self, path, output_path):
        with open(path, 'r', encoding='utf-8') as file:
            csv_data = []
            for line in file:
                    try:
                        data = json.loads(line)
                        pair_id = data['pair_id']
                        text1 = data['n1_data']['text']
                        id1 = data['n1_data']['id']
                        text2 = data['n2_data']['text']
                        id2 = data['n2_data']['id']
                        score = data['scores']['overall']
                        csv_data.append([pair_id,id1, id2,text1,text2, score])

                    except json.JSONDecodeError as e:
                        print(f"Failed to decode JSON: {e}")

            csv_data = pd.DataFrame(csv_data, columns = ['pair_id', 'id1', 'id2','text1', 'text2', 'overall'])
            csv_data.to_csv(output_path)

if __name__ == "__main__":
    RawTrainData()