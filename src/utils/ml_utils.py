import random
from transformers import XLMRobertaTokenizer
import numpy as np

class MlUtils:
    @staticmethod
    def collate_fn(batch):
        """
            TODO: Add description and type hints
        """
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", TOKENIZERS_PARALLELISM=True)

        _, _, chunks1, chunks2, combinations, values = zip(*batch)
        
        # Padding articles
        max_len1 = max(len(chunk) for article_chunks in chunks1 for chunk in article_chunks)
        max_len2 = max(len(chunk) for article_chunks in chunks2 for chunk in article_chunks)
        
        padded_chunks1 = []
        padded_chunks2 = []
        for article_chunks1, article_chunks2 in zip(chunks1, chunks2):
            padded_article_chunks1 = [chunk + [tokenizer.pad_token_id] * (max_len1 - len(chunk)) for chunk in article_chunks1]
            padded_article_chunks2 = [chunk + [tokenizer.pad_token_id] * (max_len2 - len(chunk)) for chunk in article_chunks2]
            padded_chunks1.append(padded_article_chunks1)
            padded_chunks2.append(padded_article_chunks2)
        
        return combinations, values
    
    @staticmethod
    def concatenate_texts(combinations, max_combinations=10):
        for key, value in combinations.items():
            input_ids1, input_ids2 = value # (256,) (256,)

            combined_arr = np.concatenate((input_ids1, input_ids2)).reshape(-1, 1)
             
            return combined_arr

    @staticmethod
    def create_attention_mask(input_ids):
        return np.where(input_ids != 0, 1, 0)
