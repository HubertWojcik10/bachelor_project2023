from typing import List, Tuple
from torch.utils.data import Dataset
import pandas as pd

class ArticleDataset(Dataset):
    """
        Custom Pytorch Dataset class for the articles
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
    # Assuming these are the elements you want to return
        combinations = self.df.iloc[idx]["combinations"]
        values = self.df.iloc[idx]["overall"]
        return combinations, values


