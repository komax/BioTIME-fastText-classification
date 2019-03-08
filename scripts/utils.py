#!/usr/bin/env python

from itertools import zip_longest
from collections import namedtuple

import numpy as np


ModelParams = namedtuple('ModelParams', ['dim', 'lr', 'wordNgrams', 'epoch', 'bucket'])


def parameter_combinations():
    for dim in np.linspace(start=10, stop=20, num=2, dtype=int):
        for lr in np.linspace(start=0.1, stop=1.0, num=2):
            for wordNgrams in np.linspace(start=2, stop=5, num=2, dtype=int):
                for epoch in np.linspace(start=5, stop=50, num=2, dtype=int):
                    for bucket in np.linspace(start=2_000_000, stop=10_000_000, num=2, dtype=int):
                        yield ModelParams(dim, lr, wordNgrams, epoch, bucket)


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def load_data_as_array(filename):
    with open(filename, "r") as file_data:
        content = file_data.read().splitlines()
        return np.array(content)
    raise RuntimeError("Cannot load input data")


def grouper(iterable, n, fillvalue=None):
    # from https://docs.python.org/3/library/itertools.html#itertools-recipes
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
