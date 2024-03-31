import pandas as pd
import math
import json
import datetime
import matplotlib.pyplot as plt
import os
from typing import List

class DevUtils:
    @staticmethod
    def add_overall_int_column(df: pd.DataFrame, strategy: str ="round_up") -> pd.DataFrame:
        """
            Add a new column to the dataframe with the rounded overall value
        """

        #make all columns lowercase
        df.columns = df.columns.str.lower()

        if strategy == "round_up":
            df["overall_int"] = df["overall"].apply(lambda x: math.ceil(x) if x % 1 >= 0.5 else math.floor(x))
        elif strategy == "round_down":
            df["overall_int"] = df["overall"].apply(lambda x: math.ceil(x) if x % 1 > 0.5 else math.floor(x))
        else:
            raise ValueError("Invalid strategy")

        return df
    
    @staticmethod
    def load_params(params_path: str) -> dict:
        """
            Load parameters from the given path
        """

        with open(params_path, 'r') as f:
            params = json.load(f)

        return params

    @staticmethod
    def save_losses_dict(losses: dict, model_name: str, curr_time: str) -> None:
        """
            Save the losses dictionary to the given path
        """

        save_path = f"../logs/{model_name}/{curr_time}/losses.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(losses, f)

    @staticmethod
    def plot_loss(losses: str, model_name: str, curr_time: str) -> None:
        """
            Plot the loss of a model
        """
        num_epochs = len(losses)
        fig, ax = plt.subplots(1, num_epochs, figsize=(30, 8))
        for i, (epoch, loss) in enumerate(losses.items()):
            ax[i].plot(loss, label=f"epoch {epoch+1}")
            ax[i].set_title(f"epoch {epoch+1}")
            ax[i].set_xlabel("batch")
            ax[i].set_ylabel("loss")

        plt.show()
        save_path = f"../logs/{model_name}/{curr_time}/losses.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    @staticmethod
    def save_df_with_predictions(df_reconstructed: pd.DataFrame, logits_chunks: List, logits: List, curr_time: str, model_name: str) -> None:
        """
            Save the dataframe with the predictions
        """
        save_path = f"../logs/{model_name}/{curr_time}/pred_df.csv"
        df_reconstructed["logits_chunks"] = logits_chunks
        df_reconstructed["logits"] = logits

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_reconstructed.to_csv(save_path, index=False)