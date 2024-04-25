from transformers import XLMRobertaTokenizer, XLMRobertaModel
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.chunker import Chunker
#from models.lstm.xlm_roberta import XLMRoberta
import logging
from utils.dev_utils import DevUtils
from collections import defaultdict

class LSTMOnXLMRoberta(nn.Module):
    def __init__(self, params_dict:  Dict[str, any], input_size, lstm_hidden_size, num_lstm_layers, model_name = 'xlm-roberta-base',curr_time = None, log_dir = None):
        super(LSTMOnXLMRoberta, self).__init__()
        self.xlmroberta_model = XLMRobertaModel.from_pretrained(model_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.chunker = Chunker(self.tokenizer, 512)
        self.params_dict = params_dict
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)

        parameters_to_optimize = list(self.xlmroberta_model.parameters())
        #list(self.fc.parameters()) + list(self.lstm.parameters()) +
         

        self.optimizer = torch.optim.AdamW(parameters_to_optimize, lr=1e-7, weight_decay = 0.002)
        logging.info(f"Training model with bert for 10 epochs, learning rate: lr=1e-3,  full dataset")
        self.loss_function = torch.nn.MSELoss()
        self.best_pearson = -1
        self.curr_time = curr_time
        self.log_dir = log_dir
        torch.manual_seed(30)

    def parameter_to_optimize(self):
        for param in self.xlmroberta_model.parameters():
            param.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False

    def get_data(self, path):
        df = pd.read_csv(path)
        df = df.head(5)
        return df
    
    def chunk_data(self, df):
        logging.info("Chunking data...")
        texts1, texts2, labels = df["text1"], df["text2"], df["overall"]
        input_ids = []
        for i in range(len(texts1)):
            input_id_1 = self.chunker.chunk(texts1[i])
            input_id_2 = self.chunker.chunk(texts2[i])
            input_ids.append([input_id_1, input_id_2])           
        return input_ids, labels
        
    def _manage_device(self) -> None:
        """
            Manage the device to run the model on
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xlmroberta_model.to(self.device)
        self.lstm.to(self.device)
        self.fc.to(self.device)

    def emb(self, input_ids: List[List[Tensor]], is_eval = True):
            row_emb = []
            for text in input_ids:
                #iterates through each article per row
                text_emb = []
                for chunk in text:
                    #logging.info(f"chunk: {chunk[:5]}")
                    chunk_ids = torch.tensor(chunk).unsqueeze(0)
                    attention_mask = [1 if i != 0 else 0 for i in chunk_ids.tolist()]
                    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
                    chunk_ids = chunk_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    outputs= self.xlmroberta_model(chunk_ids, attention_mask)
                    logging.info(f"outputs : {outputs.size()}")
                    #last_hidden_state= outputs.pooler_output.squeeze(0)
                    last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    #f is_eval:
                    logging.info(f"chunk: {chunk[:5]}")
                    logging.info(f"last_hidden_3: {last_hidden_state[:5]}")
                    text_emb.append(last_hidden_state)
                text_embbedings = torch.stack(text_emb)             
                row_emb.append(text_embbedings)
            return row_emb

    def pad_to_same_size(self, tensors):
        max_size = max(tensor.size(0) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            tensor.to(self.device)
            if tensor.size(0) < max_size:
                padding = torch.zeros(max_size - tensor.size(0), tensor.size(1)).to(self.device) 
                padded_tensor = torch.cat((tensor, padding), dim=0)
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
        return torch.stack(padded_tensors)

    def pearson_correlation(self, labels_val, output_val):
        """get the pearson correlation between the labels and the output of the model"""
        output_val= np.array([t.item() for t in output_val])
        return np.corrcoef(labels_val, output_val)[0][1]
    
    def predict(self, input_ids_val):
        """
            Evaluate the model """ 
        logging.info("Evaluating the model...")
        self.eval()
        with torch.no_grad():
            outputs_val = self.forward(input_ids_val, is_eval = False)

        return outputs_val   

    def forward(self, input_batch_data, is_eval = False):
        outputs = []
        for row in input_batch_data:
            row_test = self.emb(row, is_eval)
            index1, index2 = len(row_test[0]), len(row_test[1])
            row_padded = self.pad_to_same_size(row_test)                 
            row_padded = row_padded.to(self.device)
            lstm_out, _ = self.lstm(row_padded)
            lstm_out_last1 = lstm_out[0,  index1 - 1, :]
            lstm_out_last2 = lstm_out[1, index2 - 1, :]
            nn = torch.cat((lstm_out_last1, lstm_out_last2), 0)
            nn= nn.to(self.device)

            output = self.fc(nn)
            outputs.append(output)          
        return outputs

    def train_model(self,train_dff,val_df, batch_size = 4, epochs =10):
        """
            Train the model
        """
        self.parameter_to_optimize()
        losses = defaultdict(list)
       
        input_ids_val, labels_val = self.chunk_data(val_df)

        for epoch in range(epochs):
            self.train()
            train_df = train_dff.sample(frac=1).reset_index(drop=True) #shuffle 
            input_ids_train, labels_train = self.chunk_data(train_df)            
            
           

            #loss_epoch =[]
            logging.info(f"------------------------- Epoch: {epoch +1} of {epochs}-------------------------")
            for i in range(0, len(input_ids_train), batch_size):
                aggregated_logits = []
                idx=int(i/batch_size)

                input_batch_data = input_ids_train[i:i + batch_size]
                label_batch = labels_train[i:i + batch_size]
                
                outputs = self.forward(input_batch_data)
                aggregated_logits.append(outputs)
            
                labels_tensor = torch.tensor(label_batch.values, dtype=torch.float32).to(self.device)
                outputs = torch.stack(outputs).squeeze().to(self.device)
                batch_loss = self.loss_function(outputs, labels_tensor)
                losses[epoch].append(batch_loss.item())
            
                self.optimizer.zero_grad()  # Clear gradients
                batch_loss.backward()  # Backpropagation 
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()  # Update weights

                batch_pearson = self.pearson_correlation(label_batch, outputs)
                if idx % 10 == 0:
                    logging.info(f"---- Batch: {idx} of {int(len(input_ids_train)/batch_size)}----")
                    logging.info(f"Batch loss: {round(batch_loss.item(), 2)}")
                    logging.info(f"Batch pearson: {round(batch_pearson.item(), 2)}")
                    logging.info(f"true labels: {np.array([t for t in label_batch])}")
                    logging.info(f"predictions: {np.array([t.item() for t in outputs])}")

                torch.cuda.empty_cache()

            self.evaluate(input_ids_val, labels_val)
        #DevUtils.plot_loss(losses, "lstm_chunker", self.curr_time)
        #DevUtils.save_losses_dict(losses, "lstm_chunker", self.curr_time)

    def evaluate(self, input_ids_val, labels_val):
            predictions = self.predict(input_ids_val)
            logging.info(f"predictions : {[t.item() for t in predictions]}")

            predictions = torch.tensor(predictions, dtype=torch.float32)
            labels_val = torch.tensor(labels_val, dtype=torch.float32)
            eval_loss = self.loss_function(predictions, labels_val)
            logging.info(f"evaluation loss: {eval_loss}")
            eval_pearson = self.pearson_correlation(labels_val, predictions)

            logging.info(f"Current pearson: {round(eval_pearson.item(), 2)}, Best pearson: {round(self.best_pearson, 2)}")
            if eval_pearson > self.best_pearson:
                self.best_pearson = eval_pearson
                logging.info("Saving the last model...")
                torch.save(self.state_dict(), self.params_dict["lstm_save_path"])

    def run(self, train: bool = True):
        torch.autograd.set_detect_anomaly(True)
        self._manage_device()
        if train:
            logging.info("Training the model...")
            self.train_path = self.params_dict["train_data_path"]
            data_train_df= self.get_data(self.train_path)
            data_train_df = data_train_df.sample(frac=1).reset_index(drop=True) #shuffle
            train_df, val_df = data_train_df[:int(len(data_train_df) * 0.8)], data_train_df[int(len(data_train_df) * 0.8):].reset_index(drop=True)
            self.train_model(train_df, val_df)
        else:
            logging.info("Testing the model...")
            self.test_path = self.params_dict["test_data_path"]
            test_df = self.get_data(self.test_path)
            input_ids_test, labels_test = self.chunk_data(test_df)
            predictions =self.predict(input_ids_test)
            predictions = torch.tensor(predictions, dtype=torch.float32)
            labels_test = torch.tensor(labels_test, dtype=torch.float32)
            logging.info(f"predictions: {[t.item() for t in predictions]}")
            print([t.item() for t in predictions])
            #logging.info(f"labels: {[t.item() for t in labels_test]}")
            loss_test = self.loss_function(predictions, labels_test)
            logging.info(f"loss: {loss_test}")
            pearson = self.pearson_correlation(labels_test, predictions)
            logging.info(f"Pearson correlation for test data: {round(pearson.item(), 2)}")