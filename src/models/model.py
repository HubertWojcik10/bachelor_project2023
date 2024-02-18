import os
from data_preprocessing.train_data import RawTrainData
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.dev_utils import DevUtils
from typing import Tuple, List
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaConfig
import numpy as np
import time
from collections import defaultdict
from plots.plots import Plots

import logging
logging.disable(logging.WARNING)

logging.basicConfig(level=logging.INFO)

class Model:
    def __init__(self, params_dict: dict):
        self.rounding_strategy = params_dict["rounding_strategy"]
        self.model_name = params_dict["model"]
        self.batch_size = params_dict["batch_size"]
        self.epochs = params_dict["epochs"]
        self.learning_rate = params_dict["learning_rate"]
        self.train_size = params_dict["train_size"]
        self.shuffle = params_dict["shuffle"]
        self.model_save_path = params_dict["model_save_path"]
        self.summarizer_save_path = params_dict["summarizer_save_path"]

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.train_data, self.test_data = self.get_data(params_dict["train_data_path"], params_dict["test_data_path"])
        self._manage_device()
        torch.manual_seed(42)

    def _manage_device(self) -> None:
        """
            Manage the device to run the model on
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_data(self, train_data_path: str, test_data_path: str, dev: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
            Get the data from the paths
            Add the overall_int column to the dataframes
        """
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        if dev:
            train_data = train_data[:50]
            test_data = test_data[:50]

        return train_data, test_data

    def tokenize_texts(self, df: pd.DataFrame, col1: str = "text1_short", col2: str = "text2_short") -> Tuple[Tensor, Tensor]:
        """
            Tokenize the input texts and return the input_ids and attention_mask
        """

        texts1, texts2 = df[col1], df[col2]
        input_ids, attention_mask = [], []

        for idx, (t1, t2) in enumerate(zip(texts1, texts2)):
            tokenized_text = self.tokenizer(t1, t2, return_tensors="pt", padding="max_length", 
                                    truncation=True, add_special_tokens=True, max_length=512)
            input_ids.append(tokenized_text["input_ids"].tolist()[0])

            att = [1 if i != 1 else 0 for i in tokenized_text["input_ids"].tolist()[0]]
            attention_mask.append(att)

        return torch.tensor(input_ids).long(), torch.tensor(attention_mask).long()

    def split_data(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tuple[DataLoader, DataLoader]:
        """
            Split the data into training and validation sets using the DataLoader
        """

        tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
        
        train_size = int(self.train_size * len(tensor_dataset))
        val_size = len(tensor_dataset) - train_size
        train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size])

        # Define data loaders with appropriate batch size and shuffle
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)

        return train_loader, val_loader

    def predict(self, loader: DataLoader, model) -> List:
        """
            Predict the scores
        """

        model.eval()
        dev_true, dev_pred = [], []

        for idx, (ids, att, val) in enumerate(loader):
            ids, att, val = ids.to(self.device), att.to(self.device), val.to(self.device)
            with torch.no_grad():
                output = model(input_ids=ids, attention_mask=att)
                logits = output.logits

                dev_true.extend(val.cpu().numpy().tolist())
                dev_pred.extend(logits.cpu().flatten().numpy().tolist())

        cur_pearson = np.corrcoef(dev_true, dev_pred)[0][1]
        return dev_true, dev_pred, cur_pearson

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str) -> List:
        """
            Train the model
        """

        best_pearson = -1.0
        total_loss = 0
        losses = defaultdict(list)
        self.model.train()

        for epoch in range(self.epochs):
            start_time = time.time()
            #logging.info(f"Epoch {epoch+1} of {self.epochs}")
            print(f"\n{'-'*25} Epoch {epoch+1} of {self.epochs} {'-'*25}")

            for idx, (ids, att, val) in enumerate(train_loader):
                ids, att, val = ids.to(self.device), att.to(self.device), val.to(self.device)

                outputs = self.model(input_ids=ids, attention_mask=att, labels=val)
                loss, logits = outputs[:2]

                losses[epoch].append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #scheduler.step()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if idx % 5 == 0:
                    print(f"\nbatch {idx+1} of {len(train_loader)}")
                    print(f"loss: {loss.item():.2f}\n")
                    #print(f"logits: {logits}")

            print("starting validation...")
            dev_true, dev_pred, cur_pearson = self.predict(val_loader, self.model)

            print("Current dev pearson is {:.4f}, best pearson is {:.4f}".format(cur_pearson, best_pearson))
            if cur_pearson > best_pearson:
                best_pearson = cur_pearson
                print("Saving the model...")
                torch.save(self.model.state_dict(), save_path)

            print("Time costed : {}s \n".format(round(time.time() - start_time, 3)))

        Plots().plot_loss(losses)
        #return losses
