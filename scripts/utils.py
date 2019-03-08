#!/usr/bin/env python
import numpy as np

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def load_data_as_array(filename):
    with open(filename, "r") as file_data:
        content = file_data.read().splitlines()
        return np.array(content)
    raise RuntimeError("Cannot load input data")
