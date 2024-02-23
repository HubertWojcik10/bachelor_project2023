from models.baseline.baseline import Baseline
from models.text_summarizer.text_summarizer import TextSummarizer
import argparse
from utils.dev_utils import DevUtils
import logging
import os
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run models with optional training.")
    parser.add_argument("model", type=int, choices=range(1, 5), help="Model selection (1-4)")
    parser.add_argument("--train", action="store_true", help="Train the selected model")

    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    args = parser.parse_args()

    model_names = {1: "baseline", 2: "text_summarizer", 3: "combination_chunker", 4: "lstm_chunker"}

    params = DevUtils.load_params("config.json")

    model_name = model_names[args.model]

    log_dir = f"{params['log_dir']}/{model_name}/{curr_time}/model_logs"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    logging.basicConfig(filename=log_dir, filemode="w",
                format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
                level=logging.DEBUG)

    logging.info(f"Params:\n{params}")
    dev = True

    if args.model == 1:
        baseline = Baseline(params_dict=params, dev=dev, curr_time=curr_time)
        if args.train:
            baseline.run(train=True)
        else:
            baseline.run(train=False)
    elif args.model == 2:
        text_summarizer = TextSummarizer(params_dict=params, dev=dev, curr_time=curr_time)
        if args.train:
            text_summarizer.run(train=True)
        else:
            text_summarizer.run(train=False)
    elif args.model == 3:
        pass
    elif args.model == 4:
        pass
