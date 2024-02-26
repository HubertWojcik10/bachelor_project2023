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

    def __getitem__(self, idx) -> Tuple[str, str, List[List[int]], List[List[int]], List[Tuple[int, int]]]:
        article1 = self.df.iloc[idx]["text1"]
        article2 = self.df.iloc[idx]["text2"]
        chunks1 = self.df.iloc[idx]["chunks1"]
        chunks2 = self.df.iloc[idx]["chunks2"]
        combinations = self.df.iloc[idx]["combinations"]

        return article1, article2, chunks1, chunks2, combinations
