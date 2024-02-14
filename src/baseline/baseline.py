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

import logging
logging.disable(logging.WARNING)

logging.basicConfig(level=logging.INFO)

class Baseline:
    def __init__(self, params_dict: dict):
        self.rounding_strategy = params_dict["rounding_strategy"]
        self.model_name = params_dict["model"]
        self.batch_size = params_dict["batch_size"]
        self.epochs = params_dict["epochs"]
        self.learning_rate = params_dict["learning_rate"]
        self.train_size = params_dict["train_size"]
        self.shuffle = params_dict["shuffle"]
        self.model_save_path = params_dict["model_save_path"]

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

    def get_data(self, train_data_path: str, test_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
            Get the data from the paths
            Add the overall_int column to the dataframes
        """
        train_data = pd.read_csv(train_data_path)
        train_data = DevUtils.add_overall_int_column(train_data, self.rounding_strategy)
        test_data = pd.read_csv(test_data_path)
        test_data = DevUtils.add_overall_int_column(test_data, self.rounding_strategy)

        return train_data, test_data

    def shorten_text(self, text: str) -> Tensor:
        """
            Tokenize the input text and shorten it to 256 tokens. Then, return decoded text
        """

        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False, max_length = None)

        if tokenized_text["input_ids"].shape[1] > 256:
            shorten_ids =  tokenized_text["input_ids"][:, :200].tolist()[0] + tokenized_text["input_ids"][:, -54:].tolist()[0]
        else:
            shorten_ids = tokenized_text["input_ids"].tolist()[0] + [self.tokenizer.pad_token_id] * (254 - tokenized_text["input_ids"].shape[1])
        return self.tokenizer.decode(shorten_ids) 

    def tokenize_texts(self, df: pd.DataFrame, col1: str = "text1_short", col2: str = "text2_short") -> Tuple[Tensor, Tensor]:
        """
            Tokenize the input texts and return the input_ids and attention_mask
        """

        #logging.info("Tokenizing the texts...")

        texts1, texts2 = df[col1], df[col2]
        input_ids, attention_mask = [], []

        for idx, (t1, t2) in enumerate(zip(texts1, texts2)):
            tokenized_text = self.tokenizer(t1, t2, return_tensors="pt", padding="max_length", 
                                    truncation=True, add_special_tokens=True, max_length=512)
            input_ids.append(tokenized_text["input_ids"].tolist()[0])
            attention_mask.append(tokenized_text["attention_mask"].tolist()[0])

        #logging.info("Tokenization complete")

        return torch.tensor(input_ids).long(), torch.tensor(attention_mask).long()

    def split_data(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor, tensor_dataset: TensorDataset) -> Tuple[DataLoader, DataLoader]:
        """
            Split the data into training and validation sets using the DataLoader
        """

        #logging.info("Splitting the data and creating the data loaders...")
        print("Splitting the data and creating the data loaders...")

        dataset = TensorDataset(input_ids, attention_mask, labels)
        
        train_size = int(self.train_size * len(tensor_dataset))
        val_size = len(tensor_dataset) - train_size
        train_dataset, val_dataset = random_split(tensor_dataset, [train_size, val_size])

        # Define data loaders with appropriate batch size and shuffle
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)

        return train_loader, val_loader

    def predict(self, loader: DataLoader) -> List:
        """
            Predict the scores
        """

        self.model.eval()
        dev_true, dev_pred = [], []

        for idx, (ids, att, val) in enumerate(loader):
            ids, att, val = ids.to(self.device), att.to(self.device), val.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=ids, attention_mask=att, labels=val)
                loss, logits = outputs[:2]

                dev_true.extend(val.cpu().numpy().tolist()[0])
                dev_pred.extend(logits.cpu().numpy().tolist()[0])

        return dev_true, dev_pred

    def train(self, loader: DataLoader):
        """
            Train the model
        """

        #logging.info("Training the model...")
        print(f"batch size: {self.batch_size}, data size: {len(loader)}")

        best_pearson = -1.0
        total_loss = 0
        losses = []
        self.model.train()

        for epoch in range(self.epochs):
            start_time = time.time()
            #logging.info(f"Epoch {epoch+1} of {self.epochs}")
            print(f"{'-'*10} Epoch {epoch+1} of {self.epochs} {'-'*10}")

            for idx, (ids, att, val) in enumerate(loader):
                ids, att, val = ids.to(self.device), att.to(self.device), val.to(self.device)

                outputs = self.model(input_ids=ids, attention_mask=att, labels=val)
                loss, logits = outputs[:2]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #scheduler.step()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if idx % 10 == 0:
                    print("average training loss: {0:.2f}".format(total_loss / (idx+1)))
                    print("current loss:", loss.item())

                    #print(f"logits: {logits}")

            print("starting validation...")
            dev_true, dev_pred = self.predict(loader)
            cur_pearson = np.corrcoef(dev_true, dev_pred)[0][1]

            if cur_pearson > best_pearson:
                best_pearson = cur_pearson
                torch.save(self.model.state_dict(), self.model_save_path)

            print("Current dev pearson is {:.4f}, best pearson is {:.4f}".format(cur_pearson, best_pearson))
            print("Time costed : {}s \n".format(round(time.time() - start_time, 3)))
            
            # store the loss value for plotting the learning curve.
            avg_train_loss = total_loss / len(loader)
            print("average training loss: {0:.2f}".format(avg_train_loss))
            losses.append(avg_train_loss)

        return losses

    def run(self):
        """
            Run the model
        """
 
        print("Shortening the texts...")
        self.train_data["text1_short"] = self.train_data["text1"].apply(self.shorten_text)
        self.train_data["text2_short"] = self.train_data["text2"].apply(self.shorten_text)

        print("Tokenizing the texts...")
        input_ids, attention_mask = self.tokenize_texts(self.train_data)
        score = torch.tensor(self.train_data["overall"]).float()

        print("Splitting the data and creating the data loaders...")
        tensor_dataset = TensorDataset(input_ids, attention_mask, score.view(-1, 1))
        train_loader, val_loader = self.split_data(input_ids, attention_mask, score, tensor_dataset)

        print("Training the model...")
        losses = self.train(train_loader)
        print(losses)
        print("Training complete")

        return losses
