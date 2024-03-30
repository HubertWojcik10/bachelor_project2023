import random
from transformers import XLMRobertaTokenizer
import numpy as np

class MlUtils:
    @staticmethod
    def create_attention_mask(input_ids):
        return np.where(input_ids != 0, 1, 0)
