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
    
    def pad_to_same_size(self, tensors):
        max_size = max(tensor.size(0) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            if tensor.size(0) < max_size:
                padding = torch.zeros(max_size - tensor.size(0), tensor.size(1))
                padded_tensor = torch.cat((tensor, padding), dim=0)
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
        return torch.stack(padded_tensors)
    
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
                    outputs= self.model(chunk_ids, attention_mask)
                    last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    text_emb.append(last_hidden_state)
                text_emb = torch.stack(text_emb)             
                row_emb.append(text_emb)
            row_emb = self.pad_to_same_size(row_emb)
            output.append(row_emb)
        return output