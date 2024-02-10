from baseline.baseline import Baseline
import argparse
from utils.dev_utils import DevUtils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    params = DevUtils.load_params("config.json")

    baseline = Baseline(params_dict=params)
    baseline.run()