import sys
from experiments import ExperimentCLUST2D

from lsdmtrack import LSDMTracker

if __name__ == "__main__":

    test_list_root = "path/to/TestList.txt"
    model_pt = "path/to/ckpt"
    result_dir = "path/to/results/folder"
    lsdmtracker = LSDMTracker(model_path=model_pt)
    e = ExperimentCLUST2D(test_list_root, model_pt, result_dir=result_dir)
    e.run(lsdmtracker)
