from models.model import Model
import pandas as pd
from typing import Tuple
from transformers import pipeline
from torch import Tensor
import torch

class TextSummarizer(Model):
    def __init__(self, params_dict):
        super().__init__(params_dict)
        self.params_dict = params_dict
        self.summarizer = pipeline("summarization", model="Falconsai/text_summarization")

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
        if train:
            train_data, test_data = self.get_data(self.params_dict["summary_train_data_path"], self.params_dict["summary_test_data_path"])

            print("Tokenizing the texts...")
            input_ids, attention_mask = self.tokenize_texts(train_data, col1="summary1", col2="summary2")
            score = torch.tensor(self.train_data["overall"]).float()

            print("Splitting the data...")
            train_loader, val_loader = self.split_data(input_ids, attention_mask, score)

            print("Training the model...")
            self.train(train_loader, val_loader, self.summarizer_save_path)
        else:
            model = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
            model.load_state_dict(torch.load(self.summarizer_save_path))
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            print("Testing the model...")
            test_data = self.summarize(test_data)
            test_input_ids, test_attention_mask = self.tokenize_texts(test_data, col1="summary1", col2="summary2")
            test_score = torch.tensor(test_data["overall"]).float()

            test_loader = DataLoader(TensorDataset(test_input_ids, test_attention_mask, test_score), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
            dev_true, dev_pred, cur_pearson = self.predict(test_loader, model)

            print(f"Test Pearson: {cur_pearson}")