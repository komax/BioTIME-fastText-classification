#!/usr/bin/env python

from itertools import zip_longest
from collections import namedtuple

import numpy as np


ModelParams = namedtuple('ModelParams', ['dim', 'lr', 'wordNgrams', 'epoch', 'bucket'])
ParamRange = namedtuple('ParamRange', ['start', 'stop', 'num'])
ResultScore = namedtuple('Resultscore', ['precision', 'recall', 'f1score'])

DEFAULT_PARAMETER_SPACE = ModelParams(
    dim=ParamRange(start=10, stop=20, num=2),
    lr=ParamRange(start=0.1, stop=1.0, num=3),
    wordNgrams=ParamRange(start=2, stop=5, num=2),
    epoch=ParamRange(start=5, stop=50, num=2),
    bucket=ParamRange(start=2_000_000, stop=10_000_000, num=2)
)


def parameter_combinations(parameter_space=None):
    if not parameter_space:
        parameter_space = DEFAULT_PARAMETER_SPACE
    for dim in np.linspace(*parameter_space.dim, dtype=int):
        for lr in np.linspace(*parameter_space.lr):
            for wordNgrams in np.linspace(*parameter_space.wordNgrams, dtype=int):
                for epoch in np.linspace(*parameter_space.epoch, dtype=int):
                    for bucket in np.linspace(*parameter_space.bucket, dtype=int):
                        yield ModelParams(dim, lr, wordNgrams, epoch, bucket)


def f1_score(precision, recall):
    denominator = precision + recall
    if denominator == 0:
        return 0.0
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
