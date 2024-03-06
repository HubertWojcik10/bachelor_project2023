from typing import List, Tuple
import numpy as np
from collections import defaultdict
import pandas as pd

class Chunker:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token_id = self.tokenizer.sep_token_id

    def _chunk(self, text: str) -> np.ndarray:
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True, max_length=None)
        input_ids = np.array(tokenized_text["input_ids"].tolist()[0])

        # calculate the number of chunks
        num_chunks = len(input_ids) // self.max_length + (1 if len(input_ids) % self.max_length != 0 else 0)

        # calculate the total length after padding
        total_length = num_chunks * self.max_length

        # pad input_ids with 1s if needed to make its length a multiple of max_length
        if len(input_ids) < total_length:
            padding_size = total_length - len(input_ids)
            padding = np.ones(padding_size, dtype=np.int32)
            input_ids = np.concatenate((input_ids, padding))

        # split the input_ids into chunks
        chunks = np.array_split(input_ids, num_chunks)

        return np.array(chunks)

    def create_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Create chunks from the texts and add them to the dataframe
        """
        df["chunks1"] = df["text1"].apply(self._chunk)
        df["chunks2"] = df["text2"].apply(self._chunk)
        return df

    def create_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Create combinations of chunks and add them to the dataframe
        """
        
        sep_token_array = np.array([self.sep_token_id])

        combined = []
        for row_idx, row in df.iterrows():
            combinations = defaultdict(np.ndarray)
            chunks1 = row["chunks1"] # e.g. (2, 256)
            chunks2 = row["chunks2"] # e.g. (3, 256)
            
            for i, chunk1 in enumerate(chunks1):
                for j, chunk2 in enumerate(chunks2):
                    if chunk1[-1] != self.sep_token_id:
                        chunk1 = np.concatenate((chunk1, sep_token_array))

                    combinations[(row_idx, i, j)] = np.concatenate((chunk1, chunk2))
            combined.append(combinations)

        df["combinations"] = combined
        return df

    
    
