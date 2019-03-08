#! /usr/bin/env python

import pandas as pd
import fastText

from utils import f1_score

ELEMS = ['dim', 'lr', 'wordNgrams', 'epoch', 'bucket']


def cross_validate_model(model_parameters, cross_validation_sets):
    f1_scores = []
    for train_file, valid_file in cross_validation_sets:
        classifier = fastText.train_supervised(train_file,
            dim=model_parameters.dim, lr=model_parameters.lr,
            wordNgrams=model_parameters.wordNgrams, minCount=1,
            bucket=model_parameters.bucket, epoch=model_parameters.epoch)
        _, precision, recall = classifier.test(valid_file)
        f1_scores.append(f1_score(precision, recall))
    mean_f1 = sum(f1_scores) / len(f1_scores)
    return mean_f1


def test_model(model_parameters, train_data_file, test_data_file):
    classifier = fastText.train_supervised(train_data_file,
            dim=model_parameters.dim, lr=model_parameters.lr,
            wordNgrams=model_parameters.wordNgrams, minCount=1,
            bucket=model_parameters.bucket, epoch=model_parameters.epoch)
    _, precision, recall = classifier.test(test_data_file)
    return f1_score(precision, recall)


def eval_model_parameters(params_csv, cv_sets):
    for params in params_csv.itertuples(index=True, name='ModelParams'):
        f1_score = cross_validate_model(params, cv_sets)
        params_csv.at[params.Index, 'f1_train_mean'] = f1_score


def eval_test_data(params_csv, train_data, test_data):
    for params in params_csv.itertuples(index=True, name='ModelParams'):
        f1_score = test_model(params, train_data, test_data)
        params_csv.at[params.Index, 'f1_test'] = f1_score

params_csv = pd.read_csv(snakemake.input.params_csv)
params_csv['f1_train_mean'] = None
params_csv['f1_test'] = None

cv_sets = list(zip(snakemake.input.train_data, snakemake.input.valid_data))


eval_model_parameters(params_csv, cv_sets)
eval_test_data(params_csv, snakemake.input.train_file, snakemake.input.test_file)

# f1_scores = []
# for train_file, valid_file in cv_sets:
#     classifier = fastText.train_supervised(train_file,
#     dim=10, lr=1.5, wordNgrams=3, minCount=1, bucket=10_000_000,
#     epoch=50, thread=4)

#     _, precision, recall = classifier.test(valid_file)
#     f1 = f1_score(precision, recall)
#     f1_scores.append(f1)
    
# mean_f1 = sum(f1_scores) / snakemake.params.kfold
# params_csv['f1_train_mean'] = mean_f1
#print(mean_f1)

params_csv.to_csv(snakemake.output.result, index=False)