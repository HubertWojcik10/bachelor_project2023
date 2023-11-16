import json
import csv 
import pandas as pd

# Specify the path to your JSON file
train_path = '../../data/train/train.jsonl'

# Open the JSON file for reading
def reader_csv(path, output_path):
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
     reader_csv(train_path, output_path = '../../data/train/train_data.csv')