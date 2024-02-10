import pandas as pd
import math
import json

class DevUtils:
    @staticmethod
    def add_overall_int_column(df: pd.DataFrame, strategy: str ="round_up") -> pd.DataFrame:
        """
            Add a new column to the dataframe with the rounded overall value
            Args:
                df: pd.DataFrame
                strategy: str - "round_up" or "round_down"
                    param to decide if we include .5 in the upper or lower integer
            Returns:
                df: pd.DataFrame
        """

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
            Args:
                params_path: str
            Returns:
                params: dict
        """

        with open(params_path, 'r') as f:
            params = json.load(f)

        return params

