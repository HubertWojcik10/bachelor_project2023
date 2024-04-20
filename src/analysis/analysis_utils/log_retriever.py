import pandas as pd
import os

class LogRetreiver:
    def __init__(self, test_path="../../../data/test/merged_test_data.csv", model="baseline"):
        self.test_df = pd.read_csv(test_path)
        self.model = model
        self.logs_dir = f"../../../logs/{model}/"

    def get_logs(self):
        """
            Get the model_logs and true_pred.csv files from the logs directory
        """

        # get the logs directories
        logs = [d for d in os.listdir(self.logs_dir) if os.path.isdir(os.path.join(self.logs_dir, d))]

        model_logs, pred_dfs = [], []
        for log in logs:
            # find the model_logs file and open it 
            log_file = os.path.join(self.logs_dir, log, "model_logs")
            with open(log_file, "r") as f:
                model_logs.append(f.read())

            # find the csv file with the true and predicted values
            csv_file = os.path.join(self.logs_dir, log, "true_pred.csv")
            df = pd.read_csv(csv_file)

            if self.model != "chunk_combinations":
                df["pair_id"] = df["id1"].astype(str) + "_" + df["id2"].astype(str)        
            pred_dfs.append(df)

        return model_logs, pred_dfs