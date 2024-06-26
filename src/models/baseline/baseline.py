from models.model import Model
import pandas as pd
from typing import Tuple
from transformers import pipeline
from torch import Tensor
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import XLMRobertaForSequenceClassification
import logging

class Baseline(Model):
    def __init__(self, params_dict, curr_time: str, log_dir: str, dev: bool = False):
        super().__init__(params_dict, log_dir)
        self.dev = dev
        self.params_dict = params_dict
        self.curr_time = curr_time
        self.log_dir = log_dir

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

    def replace_underscore_with_zero_in_pair_ids(self, df):
        """
            Replace the underscore with zero in the pair_id column (for saving the dataframe with logits)
        """
        df['pair_id'] = df['pair_id'].str.replace("_", "0")
        
        df['pair_id'] = pd.to_numeric(df['pair_id'], errors='coerce')
        
        return df
    
    def run(self, train: bool = True) -> None:
        """
            Run the model, train it if train is True
        """

        train_data, test_data = self.get_data(self.params_dict["train_data_path"], self.params_dict["test_data_path"], self.dev)

        if train:
            train_data["text1_short"] = train_data["text1"].apply(self.shorten_text)
            train_data["text2_short"] = train_data["text2"].apply(self.shorten_text)

            input_ids, attention_mask = self.tokenize_texts(train_data, col1="text1_short", col2="text2_short")
            score = torch.tensor(train_data["overall"]).float()

            train_loader, val_loader = self.split_data(input_ids, attention_mask, score)

            self.train(train_loader, val_loader, self.model_save_path, "baseline", self.curr_time)
        else:
            model = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=1)

            save_model_path = f"{self.model_save_path}_{self.batch_size}b_{self.seed}s.pth"
            model.load_state_dict(torch.load(save_model_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            logging.info("Running on test data...")
            test_data["text1_short"] = test_data["text1"].apply(self.shorten_text)
            test_data["text2_short"] = test_data["text2"].apply(self.shorten_text)
            test_input_ids, test_attention_mask = self.tokenize_texts(test_data, col1="text1_short", col2="text2_short")
            test_score = torch.tensor(test_data["overall"]).float()

            test_data = self.replace_underscore_with_zero_in_pair_ids(test_data)
            id1 = torch.tensor(test_data["id1"]).long()
            id2 = torch.tensor(test_data["id2"]).long()


            test_loader = DataLoader(TensorDataset(test_input_ids, test_attention_mask, test_score, id1, id2), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
            dev_true, dev_pred, cur_pearson = self.predict(test_loader, model, self.curr_time, validation=False)

            print(f"Test Pearson: {cur_pearson}")


