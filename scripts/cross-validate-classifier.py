#! /usr/bin/env python

import fastText

def load_data(filename):
    with open(filename, "r") as file_data:
        return file_data.read().splitlines()
    raise RuntimeError("Cannot load input data")


def f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

cv_sets = zip(snakemake.input.train_data, snakemake.input.valid_data)
f1_scores = []
for train_file, valid_file in cv_sets:
    classifier = fastText.train_supervised(train_file,
    dim=10, lr=1.5, wordNgrams=3, minCount=1, bucket=10_000_000,
    epoch=50, thread=4)

    _, precision, recall = classifier.test(valid_file)
    f1 = f1_score(precision, recall)
    f1_scores.append(f1)
    
mean_f1 = sum(f1_scores) / snakemake.params.kfold
print(mean_f1)