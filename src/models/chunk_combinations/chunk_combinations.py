import pandas as pd
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaConfig, pipeline
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from typing import List, Tuple, Dict
import time
from sklearn.model_selection import train_test_split
import logging
import os
from utils.dev_utils import DevUtils
from models.model import Model
from utils.combinations_chunker import Chunker
from collections import defaultdict


class ChunkCombinationsModel(Model):
    def __init__(self, params_dict:  Dict[str, any], curr_time: str, dev : bool = False) -> None:
        super().__init__(params_dict)
        self.curr_time = time.strftime("%Y%m%d-%H%M%S")
        self.chunker = Chunker(self.tokenizer, 255)
        self.sep_token_id = self.tokenizer.sep_token_id
        self.params_dict = params_dict
        self.dev = dev
        self.curr_time = curr_time
        self.model_name = "chunk_combinations"

    def split_data(self, df: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
            Split the data into train and test
        """

        # select the columns and split to train and validation
        df = df[["combinations", "overall"]]
        train_df, val_df = train_test_split(df, test_size=(1- self.train_size), random_state=42, shuffle=False)

        # split the train and validation data into batches
        batch_num = np.ceil(len(train_df) / self.batch_size)
        train_batched_data = np.array_split(train_df, batch_num)
        val_batched_data = np.array_split(val_df, batch_num)

        return train_batched_data, val_batched_data
    
    def train(self, train_batched_data: List[pd.DataFrame], val_batched_data: List[pd.DataFrame], save_path: str) -> None:
        """
            Train the model
        """
        best_pearson = -1.0
        losses = defaultdict(list)
        self.model.train()

        for epoch in range(self.epochs):
            logging.info(f"{'-'*25} Epoch {epoch+1} of {self.epochs} {'-'*25}")
            start_time = time.time()

            # iterate through the batches
            for _, batch in enumerate(train_batched_data):
                combinations_list, labels_list = [], []
                labels = batch["overall"].values

                # iterate through the rows in the batch
                for label, combinations in zip(labels, batch["combinations"]):
                    # iterate through the combinations in the row
                    for key, combination in combinations.items():
                        labels_list.append(float(label))

                        combinations_list.append(torch.tensor(combination, dtype=torch.float))

                # convert the lists to tensors and move them to the device
                ids = torch.stack(combinations_list).long()
                labels = torch.tensor(labels_list, dtype=torch.float).float()
                ids, labels = ids.to(self.device), labels.to(self.device)

                # forward pass
                outputs = self.model(ids, labels=labels)
                loss, logits = outputs[:2]
                print(loss)

                losses[epoch].append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.validate(save_path, start_time, val_batched_data, best_pearson)

        # plot the loss
        DevUtils.plot_loss(losses, self.model_name, self.curr_time)

        # save the losses dictionary to a json file
        DevUtils.save_losses_dict(losses, self.model_name, self.curr_time)

    def validate(self, save_path: str, start_time: float, val_data, best_pearson: float = -1.0) -> None:
        """
            Validate the model
        """
        logging.info("Starting validation...")
        print("starting validation...")
        dev_true, dev_pred, cur_pearson = self.predict(val_data, self.model)

        logging.info(f"Current dev pearson is {cur_pearson:.4f}, best pearson is {best_pearson:.4f}")

        if cur_pearson > best_pearson:
            best_pearson = cur_pearson
            print("Saving the model...")
            torch.save(self.model.state_dict(), save_path)

        logging.info(f"Time costed : {round(time.time() - start_time, 3)}s")

    def predict(self, data, model) -> Tuple[List, List, float]:
        """
            Predict the scores
        """
        logging.info("Starting prediction...")
        model.eval()
        dev_true, dev_pred = [], []

        # iterate through the batches
        for _, batch in enumerate(data):
            with torch.no_grad():
                combinations_list, labels_list = [], []
                labels = batch["overall"].values

                # iterate through the rows in the batch
                for label, combinations in zip(labels, batch["combinations"]):

                    # iterate through the combinations in the row
                    for key, combination in combinations.items():
                        labels_list.append(float(label))

                        combinations_list.append(torch.tensor(combination, dtype=torch.float))

                # convert the lists to tensors and move them to the device
                ids, labels = torch.stack(combinations_list).long(), torch.tensor(labels_list, dtype=torch.float).float()
                ids, labels = ids.to(self.device), labels.to(self.device)

                # forward pass
                outputs = model(ids, labels=labels)
                _, logits = outputs[:2]

                # append the true and predicted values to the lists
                dev_true.extend(labels.cpu().detach().numpy())
                dev_pred.extend(logits.cpu().detach().numpy().flatten())

        # calculate the pearson correlation
        curr_pearson = np.corrcoef(dev_true, dev_pred)[0][1]
        logging.info(f"Finished prediction with pearson corr: {curr_pearson:.4f}")
        
        return dev_true, dev_pred, curr_pearson


    def run(self, train: bool = True) -> None:
        """
            Run the model   
        """
        train_data, test_data = self.get_data(self.params_dict["train_data_path"], self.params_dict["test_data_path"])
        
        if self.dev:
            train_data = train_data[:40]
            test_data = test_data[:40]

        if train:
            train_data = self.chunker.create_chunks(train_data)
            train_data = self.chunker.create_combinations(train_data)
            train_batched_data, val_batched_data = self.split_data(train_data)

            self.train(train_batched_data, val_batched_data, self.params_dict["chunk_combinations_save_path"])
        else:
            pass