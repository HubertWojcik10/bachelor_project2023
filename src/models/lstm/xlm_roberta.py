from transformers import XLMRobertaTokenizer, XLMRobertaModel
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/germa/thesis/bachelor_project2023/src/utils')
from chunker import Chunker
import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple
from torch.utils.data import TensorDataset, DataLoader, random_split

class XLMRoberta(nn.Module):
    def __init__(self, model_name: str):
        super(XLMRoberta, self).__init__()
        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.chunker = Chunker(self.tokenizer, 512)
        self.train_df = self.get_data('../../../data/train/train.csv')
    
    def get_data(self, path):
        train_df = pd.read_csv(path)
        train_short_df = train_df.head(1)
        return train_short_df
    
    def chunk_data(self, df ,col1 = "text1", col2 = "text2"):
        texts1, texts2 = df[col1], df[col2]
        input_ids = []
        for i in range(len(texts1)):
            input_id_1 = self.chunker.chunk(texts1[i])
            input_id_2 = self.chunker.chunk(texts2[i])
            input_ids.append([input_id_1, input_id_2])
            
        return input_ids

    def chunked_data(self):
        input_ids = self.chunk_data(self.train_df)
        return input_ids
    
    def run(self):
        input_ids = self.chunked_data()
        output = []
        for i in range(len(input_ids)):
            row_emb = []
            for text in input_ids[i]:
                chunks_emb = []
                for chunk in text:
                    print(chunk)
                    input_ids = torch.tensor(chunk).unsqueeze(0)
                    attention_mask = [1 if i != 0 else 0 for i in input_ids.tolist()]
                    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
                    outputs= self.model(input_ids, attention_mask)
                    last_hidden_state = outputs.last_hidden_state.mean(dim=1)
                    chunks_emb.append([last_hidden_state])
                row_emb.append([chunks_emb])
            output.append(row_emb)
        return output
        