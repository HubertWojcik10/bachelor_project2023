from collections import defaultdict
import pandas as pd
import os
from log_retriever import LogRetreiver
from transformers import XLMRobertaTokenizer
import warnings 
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ChunkComparer(LogRetreiver):
    def __init__(self, test_path="../../../data/test/merged_test_data.csv", model="baseline"):
        super().__init__(test_path, model)
        self.test_path = test_path
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.max_length = 255
        self.test_df_with_chunk_num = pd.read_csv(f"{self.test_path[:-4]}_chunk_num.csv")
        sns.set_style("darkgrid")

    def aggregate_results(self, pred_dfs):
        """
            Aggregate the results (from logs) from the different models
        """

        merged_dfs = []
        for df in pred_dfs:
            merged_df = pd.merge(self.test_df_with_chunk_num, df, on='pair_id')
            merged_dfs.append(merged_df)

        return merged_dfs

    def _create_chunk_df(self, merged_df):
        """
            Merge the dfs and average the correlation
        """
        chunk_buckets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        correlations = defaultdict(dict)
        merged_df["chunk_comb"] = merged_df["chunk_num1"] * merged_df["chunk_num2"]

        for chunk_bucket in chunk_buckets:
            # filter the df for the language pair
            chunk_df = merged_df[(merged_df['chunk_comb'] <= chunk_bucket) & (merged_df['chunk_comb'] > chunk_bucket - 2)]
            chunk_df.rename(columns={'overall_x': 'overall'}, inplace=True)

            if self.model != "chunk_combinations":
                corr = chunk_df['overall'].corr(chunk_df['pred'])
            else:
                corr = chunk_df['overall'].corr(chunk_df['logits'])
            
            # add the correlation and the count to the dictionary (if the correlation is not nan)
            if not pd.isna(corr):
                correlations[chunk_bucket] = {"corr": f"{corr:.2f}", "count": len(chunk_df), "chunk_buckets": (chunk_bucket - 2, chunk_bucket)}

        # sort the dictionary by the correlation
        correlations = dict(sorted(correlations.items(), key=lambda item: item[1]["count"], reverse=True))

        #create a table of the correlations and counts
        correlations_df = pd.DataFrame(correlations).T
        correlations_df.index.name = 'chunk_bucket'

        return correlations_df
    
    def create_aggregated_df(self, merged_dfs):
        """
            Create a df with the aggregated results
        """
        current_df = self._create_chunk_df(merged_dfs[0])
        current_df.rename(columns={"corr": f"corr_{0}"}, inplace=True)

        for idx, df in enumerate(merged_dfs[1:]):
            chunk_df = self._create_chunk_df(df)
            chunk_df.rename(columns={"corr": f"corr_{idx+1}"}, inplace=True)
            chunk_df.drop(["count", "chunk_buckets"], axis=1, inplace=True)
            current_df = pd.merge(current_df, chunk_df, on="chunk_bucket")
            #current_df.drop("count", axis=1, inplace=True)

        
        corr_cols_num = len(current_df.columns) - 2
        corr_cols = [f"corr_{i}" for i in range(corr_cols_num)]
        for col in corr_cols:
            current_df[col] = pd.to_numeric(current_df[col], errors="coerce")

        current_df["avg_corr"] = current_df[[f"corr_{i}" for i in range(len(corr_cols))]].mean(axis=1)

        current_df.drop([f"corr_{i}" for i in range(len(corr_cols))], axis=1, inplace=True)

        return current_df
        


    def _add_chunk_num(self, df):
        """
            Add the chunk number to the df
            Note: only need to run this once
        """
        df["chunk_num1"] = df["text1"].apply(lambda x: len(self._chunk(x)))
        df["chunk_num2"] = df["text2"].apply(lambda x: len(self._chunk(x)))

        #save the dataframe in test_path
        df.to_csv(f"{self.test_path[:-4]}_chunk_num.csv", index=False)
        return df
    
    def _chunk(self, text):
        """
            Chunk the text into smaller pieces
            Note: only need to run this once
        """
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding= True, truncation=False, add_special_tokens=True, max_length= None)
        input_ids = tokenized_text["input_ids"].tolist()[0]
        chunks = [input_ids[i:i+self.max_length] for i in range(0, len(input_ids), self.max_length)]
        return chunks

    def plot_chunk_combinations_num(self, df):
        """
            Plot the number of chunk combinations
        """

        df["chunk_comb"] = df.apply(lambda x: x["chunk_num1"] * x["chunk_num2"], axis=1)
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
        # create a histogram of using seaborn for the chunk combinations
        sns.histplot(df["chunk_comb"], bins=range(0, 30, 2), alpha=0.7)
        ax.set_title("Chunk Combination Number")
        ax.set_xlabel("Number of Chunk Combinations")
        ax.set_ylabel("Frequency")
        plt.show()


    def plot_chunk_num(self, df):
        """
            Plot the chunk number
        """

        #create a histogram of the chunk number for text1 and text2 with a max of 20 chunks and a bin size of 2
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(df["chunk_num1"], bins=range(0, 20, 2), alpha=0.7)
        ax[0].set_title("Text1 Chunk Number")
        ax[0].set_xlabel("Number of Chunks")
        ax[0].set_ylabel("Frequency")

        ax[1].hist(df["chunk_num2"], bins=range(0, 20, 2), alpha=0.7)
        ax[1].set_title("Text2 Chunk Number")
        ax[1].set_xlabel("Number of Chunks")
        ax[1].set_ylabel("Frequency")

        plt.show()

    def run(self):
        model_logs, pred_dfs = self.get_logs()
        merged_dfs = self.aggregate_results(pred_dfs)
        df = self.create_aggregated_df(merged_dfs)
        return df