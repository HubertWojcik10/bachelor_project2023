from models.baseline.baseline import Baseline
from models.text_summarizer.text_summarizer import TextSummarizer
import argparse
from utils.dev_utils import DevUtils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    params = DevUtils.load_params("config.json")

    #baseline = Baseline(params_dict=params)
    #baseline.run(train=True)



    text_summarizer = TextSummarizer(params_dict=params)
    text_summarizer.run(train=True)



