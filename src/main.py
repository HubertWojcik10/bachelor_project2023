from models.baseline.baseline import Baseline
from models.text_summarizer.text_summarizer import TextSummarizer
import argparse
from utils.dev_utils import DevUtils
import logging
import os
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline and text summarizer models with optional training.")
    parser.add_argument("--baseline", action="store_true", help="Run baseline model")
    parser.add_argument("--text_summarizer", action="store_true", help="Run text summarizer model")
    parser.add_argument("--train_baseline", action="store_true", help="Train baseline model")
    parser.add_argument("--train_text_summarizer", action="store_true", help="Train text summarizer model")
    args = parser.parse_args()

    params = DevUtils.load_params("config.json")

    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(filename=f"{params['log_dir']}/model_{curr_time}", filemode="w",
                format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
                level=logging.DEBUG)

    logging.info(f"Params:\n{params}")
    dev = False

    if args.baseline:
        baseline = Baseline(params_dict=params, dev=dev)
        if args.train_baseline:
            baseline.run(train=True)
        else:
            baseline.run(train=False)

    if args.text_summarizer:
        text_summarizer = TextSummarizer(params_dict=params)
        if args.train_text_summarizer:
            text_summarizer.run(train=True)
        else:
            text_summarizer.run(train=False)
