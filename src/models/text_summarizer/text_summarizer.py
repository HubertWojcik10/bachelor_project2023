from models.model import Model
import pandas as pd
from typing import Tuple
from transformers import pipeline
from torch import Tensor
import torch
from transformers import XLMRobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

class TextSummarizer(Model):
    def __init__(self, params_dict, dev, log_dir: str, curr_time: str):
        super().__init__(params_dict, log_dir)
        self.params_dict = params_dict
        self.summarizer = pipeline("summarization", model=params_dict["summarizer_model"])
        self.curr_time = curr_time
        self.dev = dev

    def _add_pad_token_to_text(self, text: str, max_length: int=254):
        return text + " <pad>" * (max_length - len(text.split()))

    def summarize(self, df):
        df['summary1'] = df['text1'].apply(
            lambda x: self.summarizer(x, max_length=254, min_length=150, do_sample=False)[0]['summary_text'] if len(x.split()) > 254 else x).apply(lambda x: self._add_pad_token_to_text(x, 254))
        df['summary2'] = df['text2'].apply(
            lambda x: self.summarizer(x, max_length=254, min_length=150, do_sample=False)[0]['summary_text'] if len(x.split()) > 254 else x).apply(lambda x: self._add_pad_token_to_text(x, 254))

        return df

    def run(self, train: bool = True) -> None:
        """
            Run the model
        """
        train_data, test_data = self.get_data(self.params_dict["summary_train_data_path"], self.params_dict["summary_test_data_path"])
        if train:
            input_ids, attention_mask = self.tokenize_texts(train_data, col1="summary1", col2="summary2")
            score = torch.tensor(self.train_data["overall"]).float()

            train_loader, val_loader = self.split_data(input_ids, attention_mask, score)

            save_path = f"{self.summarizer_save_path}_{self.batch_size}b_{self.seed}s.pth"
            self.train(train_loader, val_loader, save_path, "text_summarizer", self.curr_time)
        else:
            model = XLMRobertaForSequenceClassification.from_pretrained(self.params_dict["model"], num_labels=1)
            save_path = f"{self.summarizer_save_path}_{self.batch_size}b_{self.seed}s.pth"
            model.load_state_dict(torch.load(save_path))
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            test_data = self.summarize(test_data)
            test_input_ids, test_attention_mask = self.tokenize_texts(test_data, col1="summary1", col2="summary2")
            test_score = torch.tensor(test_data["overall"]).float()

            test_loader = DataLoader(TensorDataset(test_input_ids, test_attention_mask, test_score), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
            dev_true, dev_pred, cur_pearson = self.predict(test_loader, model)