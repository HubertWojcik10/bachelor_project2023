from models.model import Model
import pandas as pd
from typing import Tuple
from transformers import pipeline
from torch import Tensor
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import XLMRobertaForSequenceClassification

class Baseline(Model):
    def __init__(self, params_dict):
        super().__init__(params_dict)
        self.params_dict = params_dict

    def shorten_text(self, text: str) -> Tensor:
        """
            Tokenize the input text and shorten it to 256 tokens. Then, return decoded text
        """

        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False, max_length = None)

        if tokenized_text["input_ids"].shape[1] > 254:
            shorten_ids =  tokenized_text["input_ids"][:, :200].tolist()[0] + tokenized_text["input_ids"][:, -54:].tolist()[0]
        else:
            shorten_ids = tokenized_text["input_ids"].tolist()[0] + [self.tokenizer.pad_token_id] * (254 - tokenized_text["input_ids"].shape[1])
        return self.tokenizer.decode(shorten_ids) 
    
    def run(self, train: bool = True) -> None:
        """
            Run the model, train it if train is True
        """

        train_data, test_data = self.get_data(self.params_dict["train_data_path"], self.params_dict["test_data_path"])

        if train:
            train_data["text1_short"] = train_data["text1"].apply(self.shorten_text)
            train_data["text2_short"] = train_data["text2"].apply(self.shorten_text)

            print("Tokenizing the texts...")
            input_ids, attention_mask = self.tokenize_texts(train_data, col1="text1_short", col2="text2_short")
            score = torch.tensor(train_data["overall"]).float()

            print("Splitting the data...")
            train_loader, val_loader = self.split_data(input_ids, attention_mask, score)

            print("Training the model...")
            self.train(train_loader, val_loader, self.model_save_path)
        else:
            model = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
            model.load_state_dict(torch.load(self.model_save_path))

            print("Testing the model...")
            test_data["text1_short"] = test_data["text1"].apply(self.shorten_text)
            test_data["text2_short"] = test_data["text2"].apply(self.shorten_text)
            test_input_ids, test_attention_mask = self.tokenize_texts(test_data, col1="text1_short", col2="text2_short")
            test_score = torch.tensor(test_data["overall"]).float()

            test_loader = DataLoader(TensorDataset(test_input_ids, test_attention_mask, test_score), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
            dev_true, dev_pred, cur_pearson = self.predict(test_loader, model)

            print(f"Test Pearson: {cur_pearson}")


