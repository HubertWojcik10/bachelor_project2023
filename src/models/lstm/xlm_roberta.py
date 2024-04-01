from transformers import XLMRobertaTokenizer, XLMRobertaModel
import pandas as pd
import numpy as np
import sys
from utils.chunker import Chunker
import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple

class XLMRoberta(nn.Module):
    def __init__(self, model_name: str):
        super(XLMRoberta, self).__init__()
        self.model = XLMRobertaModel.from_pretrained(model_name)
        torch.manual_seed(42)
        
        
    def run(self, input_ids: List[List[List[Tensor]]]):
        output = []
        for i in range(len(input_ids)):
            #iterates through the rows
            row_emb = []
            for text in input_ids[i]:
                #iterates through each article per row
                text_emb = []
                for chunk in text:
                    chunk_ids = torch.tensor(chunk).unsqueeze(0)
                    attention_mask = [1 if i != 0 else 0 for i in chunk_ids.tolist()]
                    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
                    chunk_ids = chunk_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    outputs= self.model(chunk_ids, attention_mask)
                    last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    text_emb.append(last_hidden_state)
                text_emb = torch.stack(text_emb)             
                row_emb.append(text_emb)
            output.append(row_emb)
        return output
