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

class XLMRoberta(nn.Module):
    def __init__(self, model_name: str):
        super(XLMRoberta, self).__init__()
        self.model = XLMRobertaModel.from_pretrained(model_name)
  
    def run(self, input_ids: List[List[List[Tensor]]]):
        output = []
        for i in range(len(input_ids)):
            #iterates through the rows
            row_emb = []
            for text in input_ids[i]:
                #iterates through each article per row
                chunks_emb = []
                for chunk in text:
                    chunk_ids = torch.tensor(chunk).unsqueeze(0)
                    attention_mask = [1 if i != 0 else 0 for i in chunk_ids.tolist()]
                    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
                    outputs= self.model(chunk_ids, attention_mask)
                    last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    chunks_emb.append(last_hidden_state)
                chunks_emb = torch.stack(chunks_emb).unsqueeze(0)                
                row_emb.append(chunks_emb)
            output.append(row_emb)
        return output