import random
from transformers import XLMRobertaTokenizer

class MlUtils:
    @staticmethod
    def collate_fn(batch):
        """
            TODO: Add description and type hints
        """
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", TOKENIZERS_PARALLELISM=True)

        articles1, articles2, chunks1, chunks2, combinations = zip(*batch)
        
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
        
        return articles1, articles2, padded_chunks1, padded_chunks2, combinations
    
    @staticmethod
    def concatenate_texts(text1_chunks, text2_chunks, max_combinations=10):
        concatenated_texts = []
        for chunk1 in text1_chunks:
            for chunk2 in text2_chunks:
                concatenated_texts.append(chunk1 + chunk2)

        if len(concatenated_texts) > max_combinations:
            concatenated_texts = random.sample(concatenated_texts, max_combinations)
        return concatenated_texts