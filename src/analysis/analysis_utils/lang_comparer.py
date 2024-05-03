from collections import defaultdict
import pandas as pd
import os
from log_retriever import LogRetreiver

class LangComparer(LogRetreiver):
    def __init__(self, test_path="../../../data/test/merged_test_data.csv", model="baseline"):
        super().__init__(test_path, model)

    def aggregate_results(self, pred_dfs):
        """
            Aggregate the results from the different models
        """
        # create a dictionary to store the results
        results = defaultdict(list)

        merged_dfs = []
        for df in pred_dfs:
            merged_df = pd.merge(self.test_df, df, on='pair_id')

            # create a new column with the language pair
            merged_df["lang"] = merged_df["lang1"].astype(str) + "_" + merged_df["lang2"].astype(str)

            if self.model != "chunk_combinations":
                merged_df.drop(['id1_y', 'id2_y', 'true'], axis=1, inplace=True)
                merged_df.rename(columns={'id1_x': 'id1', 'id2_x': 'id2'}, inplace=True)
            else:
                merged_df.drop(['overall_y'], axis=1, inplace=True)
                merged_df.rename(columns={'overall_x': 'overall'}, inplace=True)

            merged_dfs.append(merged_df)

        return merged_dfs


    def _create_lang_df(self, merged_df):
        """
            Merge the dfs and average the correlation
        """
        #create a list of the unique language pairs
        lang_pairs = merged_df['lang1'] + '_' + merged_df['lang2']
        lang_pairs = lang_pairs.unique()

        correlations = defaultdict(dict)

        for lang_pair in lang_pairs:
            # filter the df for the language pair
            lang_pair_df = merged_df[merged_df['lang1'] + '_' + merged_df['lang2'] == lang_pair]

            if self.model != "chunk_combinations":
                corr = lang_pair_df['overall'].corr(lang_pair_df['pred'])
            else:
                corr = lang_pair_df['overall'].corr(lang_pair_df['logits'])
            
            # add the correlation and the count to the dictionary (if the correlation is not nan)
            if not pd.isna(corr):
                correlations[lang_pair] = {"corr": f"{corr:.2f}", "count": len(lang_pair_df)}

        # sort the dictionary by the correlation
        correlations = dict(sorted(correlations.items(), key=lambda item: item[1]["count"], reverse=True))

        #create a table of the correlations and counts
        correlations_df = pd.DataFrame(correlations).T
        correlations_df.index.name = 'lang_pair'

        return correlations_df  

    def create_aggregated_df(self, merged_dfs):
        """
            Create a df with the aggregated results
        """
        current_df = self._create_lang_df(merged_dfs[0])
        current_df.rename(columns={'corr': f'corr_{0}'}, inplace=True)
        for idx, df in enumerate(merged_dfs[1:]):
            lang_df = self._create_lang_df(df)
            
            lang_df = lang_df.drop(["count"], axis=1)
            
            lang_df.rename(columns={'corr': f'corr_{idx+1}'}, inplace=True)
            current_df = pd.merge(current_df, lang_df, on="lang_pair")

        corr_cols_num = len(current_df.columns) - 1
        corr_cols = [f"corr_{i}" for i in range(corr_cols_num)]
        
        for col in corr_cols:
            current_df[col] = pd.to_numeric(current_df[col], errors='coerce')

        current_df["mean_corr"] = current_df[corr_cols].mean(axis=1)

        current_df.drop(corr_cols, axis=1, inplace=True)
        return current_df

    def run(self):
        """
            Run the functions
        """
        model_logs, pred_dfs = self.get_logs()
        merged_dfs = self.aggregate_results(pred_dfs)
        df = self.create_aggregated_df(merged_dfs)

        return df