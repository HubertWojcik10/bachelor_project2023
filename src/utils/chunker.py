from typing import List, Tuple
import pandas as pd

class Chunker:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def chunk(self, text: str) -> List[List[int]]:
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding= True, truncation=False, add_special_tokens=True, max_length= None)
        input_ids = tokenized_text["input_ids"].tolist()[0]
        chunks = [[input_ids[i:i+self.max_length] for i in range(0, len(input_ids), self.max_length)]]
        return chunks
    