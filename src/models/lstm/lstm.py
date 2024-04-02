from transformers import XLMRobertaTokenizer, XLMRobertaModel
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.chunker import Chunker
from models.lstm.xlm_roberta import XLMRoberta


class LSTMOnXLMRoberta(nn.Module):
    def __init__(self, params_dict:  Dict[str, any], input_size, lstm_hidden_size, num_lstm_layers, model_name = 'xlm-roberta-base', train_size = 0.8, batch_size = 32, shuffle = True):
        super(LSTMOnXLMRoberta, self).__init__()
        self.xlmroberta_model = XLMRoberta(model_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.chunker = Chunker(self.tokenizer, 512)
        self.params_dict = params_dict
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)

        parameters_to_optimize = list(self.fc.parameters()) + list(self.lstm.parameters()) + list(self.xlmroberta_model.parameters())

        self.optimizer = torch.optim.AdamW(parameters_to_optimize, lr=1e-2)
        self.loss_function = nn.MSELoss()
        torch.manual_seed(42)

    def parameter_to_optimize(self):
        for param in self.xlmroberta_model.parameters():
            param.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def get_data(self, path):
        df = pd.read_csv(path)
        #df = df.head(50)
        return df
    
    def chunk_data(self, df):
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
    
    def get_embeddings(self, input_ids):
        """
            Get the embeddings from the XLM-Roberta model
        """
        outputs = self.xlmroberta_model.run(input_ids)
        return outputs

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
        output_val= np.array([t.item() for t in output_val])
        return np.corrcoef(labels_val, output_val)[0][1]
    
    def predict(self, input_ids_val, batch_size = 4):
        """
            Evaluate the model """ 
        self.eval()
        output_val = []
        with torch.no_grad():
            for i in range(0, len(input_ids_val), batch_size):
                input_batch_data = input_ids_val[i:i + batch_size]
                input_batch = self.get_embeddings(input_batch_data)
                for row in input_batch:
                    index1, index2 = len(row[0]), len(row[1])
                    row_padded = self.pad_to_same_size(row) 
                    lstm_out, _ = self.lstm(row_padded)
                    lstm_out_last1 = lstm_out[0, index1 - 1, :]
                    lstm_out_last2 = lstm_out[1, index2 - 1, :]
                    nn = torch.cat((lstm_out_last1, lstm_out_last2), 0)
                    output = self.fc(nn)
                    output_val.append(output)

        return output_val   

    def train_model(self, input_ids_train, labels_train, input_ids_val, labels_val, batch_size = 4):
        """
            Train the model
        """
        best_pearson = -1
        self.train()
        for epoch in range(5):
            print(f"Epoch: {epoch}")

            for i in range(0, len(input_ids_train), batch_size):
                print(f"Batch: {int(i/batch_size)}/{int(len(input_ids_train)/batch_size)}")
                input_batch_data = input_ids_train[i:i + batch_size]
                input_batch = self.get_embeddings(input_batch_data)
                label_batch = labels_train[i:i + batch_size]
                batch_loss = 0
                self.optimizer.zero_grad()  # Clear gradients
                outputs = []
                for row, label in zip(input_batch, label_batch):
                    index1, index2 = len(row[0]), len(row[1])
                    row_padded = self.pad_to_same_size(row) 
                    row_padded = row_padded.to(self.device)
                    lstm_out, _ = self.lstm(row_padded)
                    lstm_out_last1 = lstm_out[0, index1 - 1, :]
                    lstm_out_last2 = lstm_out[1, index2 - 1, :]
                    nn = torch.cat((lstm_out_last1, lstm_out_last2), 0)
                    nn= nn.to(self.device)
                    output = self.fc(nn)
                    outputs.append(output)
                    label=torch.tensor(label, dtype=torch.float32).view(1)
                    label = label.to(self.device)
                    loss = self.loss_function(output,label)
                    batch_loss = batch_loss + loss

                batch_loss = batch_loss/batch_size 
                batch_loss.backward()  # Backpropagation 
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()  # Update weights
                print(f"Batch loss: {batch_loss}")
                #print(max(outputs))
                batch_pearson = self.pearson_correlation(label_batch, outputs)
                print(f"Batch pearson: {batch_pearson}")

            predictions = self.predict(input_ids_val)
            #print(max(predictions))
            eval_pearson = self.pearson_correlation(labels_val, predictions)
            
            if eval_pearson > best_pearson:
                best_pearson = eval_pearson
                print("Saving the model...")
                torch.save(self.state_dict(), self.params_dict["lstm_save_path"])


    def run(self, train: bool = True):
        torch.autograd.set_detect_anomaly(True)
        self._manage_device()
        if train:
            self.train_path = self.params_dict["train_data_path"]
            train_df= self.get_data(self.train_path)
            train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle 
            train_df, val_df = train_df[:int(len(train_df) * 0.8)], train_df[int(len(train_df) * 0.8):].reset_index(drop=True)
            input_ids_train, labels_train = self.chunk_data(train_df)
            input_ids_val, labels_val = self.chunk_data(val_df)
            self.parameter_to_optimize()
            self.train_model(input_ids_train, labels_train, input_ids_val, labels_val)
        else:
            self.test_path = self.params_dict["test_data_path"]
            test_df = self.get_data(self.test_path)
            input_ids_test, labels_test = self.chunk_data(test_df)
            predictions =self.predict(input_ids_test)
            #print(predictions)
            pearson = self.pearson_correlation(labels_test, predictions)
            print(pearson)