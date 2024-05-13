import torch
import numpy as np
from typing import List, Tuple, Dict
import time
from collections import defaultdict
import pandas as pd
from utils.ml_utils import MlUtils
from transformers import XLMRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from utils.dev_utils import DevUtils
from utils.logger import Logger
from models.model import Model
from utils.combinations_chunker import Chunker



class ChunkCombinationsModel(Model):
    """
        Model nr 4: Create combinations of chunks and train the model
    """
    def __init__(self, params_dict:  Dict[str, any], curr_time: str, log_dir: str, dev : bool = False) -> None:
        super().__init__(params_dict, log_dir)
        self.curr_time = time.strftime("%Y%m%d-%H%M%S")
        self.chunker = Chunker(self.tokenizer, 255)
        self.sep_token_id = self.tokenizer.sep_token_id
        self.params_dict = params_dict
        self.dev = dev
        self.curr_time = curr_time
        self.model_name = "chunk_combinations"
        self.logger = Logger(log_dir)
        #self.random_seed = params_dict["random_seed"]
        torch.manual_seed(self.seed)

    def split_train_data(self, df: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
            Split the train data into train and validation
        """

        # select the columns and split to train and validation
        df = df[["pair_id", "combinations", "overall"]]
        train_df, val_df = train_test_split(df, test_size=(1-self.train_size), random_state=self.seed, shuffle=True)

        # split the train and validation data into batches
        batch_num = np.ceil(len(train_df) / self.batch_size)
        train_batched_data = np.array_split(train_df, batch_num)
        val_batched_data = np.array_split(val_df, batch_num)

        return train_batched_data, val_batched_data
    
    def split_test_data(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
            Split the test data into batches
        """
        df = df[["pair_id", "combinations", "overall"]]
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        batch_num = np.ceil(len(df) / self.batch_size)
        test_batched_data = np.array_split(df, batch_num)

        return test_batched_data

    
    def train(self, train_batched_data: List[pd.DataFrame], val_batched_data: List[pd.DataFrame], save_path: str) -> None:
        """
            Train the model
        """
        self.model.train()
        self.logger.log_model_info("start_train")

        criterion = torch.nn.MSELoss()
        best_pearson = -1.0
        losses = defaultdict(list)

        for epoch in range(self.epochs):
            self.logger.log_epoch_info(epoch, self.epochs)
            start_time = time.time()

            # iterate through the batches
            for idx, batch in enumerate(train_batched_data):
                aggregated_logits, labels = [], batch["overall"].values
                pair_ids = batch["pair_id"].values

                # iterate through the rows in the batch
                for label, combinations, pair_id in zip(labels, batch["combinations"], pair_ids):
                    logits_list = []

                    # iterate through the combinations in the row
                    for key, combination in combinations.items():

                        # create the attention mask
                        att_mask = MlUtils.create_attention_mask(combination)

                        # convert the lists to tensors and move them to the device
                        ids = torch.tensor(combination, dtype=torch.long).unsqueeze(0).to(self.device)
                        att = torch.tensor(att_mask, dtype=torch.float).unsqueeze(0).to(self.device)
                        
                        # get the output from the model and append the logits to the list
                        output = self.model(input_ids=ids, attention_mask=att)
                        logits_list.append(output.logits)
                        
                    aggregated_logits.append(torch.mean(torch.stack(logits_list), dim=0))

                # convert the lists to tensors and move them to the device
                labels_tensor = torch.tensor(labels, dtype=torch.float).to(self.device)
                outputs = torch.stack(aggregated_logits).squeeze().to(self.device)

                # calculate the loss
                loss = criterion(outputs, labels_tensor)

                # initiate backpropagation
                print(f"loss: {loss.item()}")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                losses[epoch].append(loss.item())
                self.logger.log_batch_info(idx, len(train_batched_data), loss)

            self.validate(save_path, start_time, val_batched_data, best_pearson)

        # plot the loss
        DevUtils.plot_loss(losses, self.model_name, self.curr_time)

        # save the losses dictionary to a json file
        DevUtils.save_losses_dict(losses, self.model_name, self.curr_time)

    def validate(self, save_path: str, start_time: float, val_data, best_pearson: float = -1.0) -> None:
        """
            Validate the model
        """
        self.logger.log_model_info("start_validation")
        _, _, cur_pearson = self.predict(val_data, self.model, test=False)

        self.logger.log_model_info("finished_validation", cur_pearson, best_pearson)

        if cur_pearson > best_pearson:
            best_pearson = cur_pearson
            self.logger.log_saving_model(save_path)
            torch.save(self.model.state_dict(), save_path)

        self.logger.log_time_cost(start_time, time.time())

    def predict(self, data, model, test: bool = True) -> Tuple[List, List, float]:
        """
            Predict the scores
        """
        self.logger.log_model_info("start_prediction")
        model.eval()
        dev_true, dev_pred = [], []
        df_reconstructed = pd.concat([pd.DataFrame(batch) for batch in data], ignore_index=True)
        logits_chunks_for_df = []
        logits_for_df = []
        

        # iterate through the batches
        for idx, batch in enumerate(data):
            self.logger.log_test_batch_info(idx, len(data))
            with torch.no_grad():
                aggregated_logits, labels = [], batch["overall"].values
                pair_ids = batch["pair_id"].values

                # iterate through the rows in the batch
                for label, combinations, pair_ids in zip(labels, batch["combinations"], pair_ids):
                    logits_list = []
                    # iterate through the combinations in the row
                    for _, combination in combinations.items():
                        att_mask = MlUtils.create_attention_mask(combination)

                        # convert the lists to tensors and move them to the device
                        ids = torch.tensor(combination, dtype=torch.long).unsqueeze(0).to(self.device)
                        att = torch.tensor(att_mask, dtype=torch.float).unsqueeze(0).to(self.device)

                        # get the output from the model and append the logits to the list
                        output = model(input_ids=ids, attention_mask=att)
                        logits_list.append(output.logits)
                    
                    prediction = torch.mean(torch.stack(logits_list), dim=0)
                    logits_for_df.append(prediction.item())
                    aggregated_logits.append(prediction)

                    logits_list_for_df = [l.tolist() for l in logits_list]
                    logits_chunks_for_df.append(logits_list_for_df)


                # convert the lists to tensors and move them to the device
                labels_tensor = torch.tensor(labels, dtype=torch.float).to(self.device)
                outputs = torch.stack(aggregated_logits).squeeze().to(self.device)

                # append the true and predicted values to the lists
                dev_true.extend(labels_tensor.cpu().detach().numpy())
                dev_pred.extend(outputs.cpu().detach().numpy().flatten())

        # calculate the pearson correlation
        curr_pearson = np.corrcoef(dev_true, dev_pred)[0][1]

        # save the dataframe with the predictions
        if test:
            DevUtils.save_df_with_predictions(df_reconstructed, logits_chunks_for_df, logits_for_df, self.curr_time, self.model_name)

        self.logger.log_model_info("finished_prediction", curr_pearson)
        
        return dev_true, dev_pred, curr_pearson


    def run(self, train: bool = True) -> None:
        """
            Run the model   
        """
        train_data, test_data = self.get_data(self.params_dict["train_data_path"], self.params_dict["test_data_path"])
        
        if self.dev:
            train_data = train_data[:15]
            test_data = test_data[:15]
            print("Running in dev mode")

        if train:
            train_data = self.chunker.create_chunks(train_data)
            train_data = self.chunker.create_combinations(train_data)
            train_batched_data, val_batched_data = self.split_train_data(train_data)

            save_path = f'{self.params_dict["chunk_combinations_save_path"][:-4]}_{self.batch_size}b_{self.seed}s'
            self.train(train_batched_data, val_batched_data, save_path)
        else:
            test_data = self.chunker.create_chunks(test_data)
            test_data = self.chunker.create_combinations(test_data)

            test_batched_data = self.split_test_data(test_data)

            model = XLMRobertaForSequenceClassification.from_pretrained(self.params_dict["model"], num_labels=1)
            model_save_path = f"{self.params_dict['chunk_combinations_save_path'][:-4]}_{self.batch_size}b_{self.seed}s.pth"
            model.load_state_dict(torch.load(model_save_path))
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            dev_true, dev_pred, cur_pearson = self.predict(test_batched_data, model)