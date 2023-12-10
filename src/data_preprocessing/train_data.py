import json
import csv 
import pandas as pd


class RawTrainData:
    def __init__(self, train_data_file_path="../../data/train/train.jsonl", output_path="../../data/train/train.csv"):
        self.train_data_file_path = train_data_file_path
        self.output_path = output_path

    def reader_csv(self):
        """
            Read the jsonl file and convert it to csv
        """
        with open(self.train_data_file_path, "r", encoding="utf-8") as file:
            csv_data = []
            for line in file:
                    try:
                        data = json.loads(line)
                        pair_id = data["pair_id"]
                        text1 = data["n1_data"]["text"]
                        id1 = data["n1_data"]["id"]
                        text2 = data["n2_data"]["text"]
                        id2 = data["n2_data"]["id"]
                        score = data["scores"]["overall"]
                        lang1 = data["n1_data"]["meta_lang"]
                        lang2 = data["n2_data"]["meta_lang"]
                        csv_data.append([pair_id,id1, id2,text1,text2, score, lang1, lang2])

                    except json.JSONDecodeError as e:
                        print(f"Failed to decode JSON: {e}")

            csv_data = pd.DataFrame(csv_data, columns = ["pair_id", "id1", "id2", "text1", "text2", "overall", "lang1", "lang2"])
            csv_data.to_csv(self.output_path)

if __name__ == "__main__":
    rtd = RawTrainData()
    rtd.reader_csv()