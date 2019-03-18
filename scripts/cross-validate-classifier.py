#! /usr/bin/env python

import pandas as pd
import fastText

from utils import f1_score

THREADS = 1

def cross_validate_model(model_parameters, cross_validation_sets):
    f1_scores = []
    for train_file, valid_file in cross_validation_sets:
        classifier = fastText.train_supervised(train_file,
            dim=model_parameters.dim, lr=model_parameters.lr,
            wordNgrams=model_parameters.wordNgrams, minCount=1,
            bucket=model_parameters.bucket, epoch=model_parameters.epoch, thread=THREADS)
        _, precision, recall = classifier.test(valid_file)
        f1_scores.append(f1_score(precision, recall))
    mean_f1 = sum(f1_scores) / len(f1_scores)
    return mean_f1


def test_model(model_parameters, train_data_file, test_data_file):
    classifier = fastText.train_supervised(train_data_file,
            dim=model_parameters.dim, lr=model_parameters.lr,
            wordNgrams=model_parameters.wordNgrams, minCount=1,
            bucket=model_parameters.bucket, epoch=model_parameters.epoch, thread=THREADS)
    _, precision, recall = classifier.test(test_data_file)
    return f1_score(precision, recall)


def eval_model_parameters(params_csv, cv_sets):
    for params in params_csv.itertuples(index=True, name='ModelParams'):
        f1_score = cross_validate_model(params, cv_sets)
        params_csv.at[params.Index, 'f1_cross_validation'] = f1_score


def eval_test_data(params_csv, train_data, test_data):
    for params in params_csv.itertuples(index=True, name='ModelParams'):
        f1_score = test_model(params, train_data, test_data)
        params_csv.at[params.Index, 'f1_test'] = f1_score


def main():
    global THREADS
    if 'snakemake' in globals():
        parameters_file_name = snakemake.input.params_csv
        cv_sets = list(zip(snakemake.input.train_data, snakemake.input.valid_data))
        train_file_name = snakemake.input.train_file
        test_file_name = snakemake.input.test_file
        result_csv = snakemake.output.result
        THREADS = snakemake.threads
    else:
        import sys
        from pathlib import Path
        _, parameters_file_name, cv_dir, result_csv, THREADS = sys.argv
        cv_path = Path(cv_dir)
        train_file_name = str(next(cv_path.glob('*.train')))
        test_file_name = str(next(cv_path.glob('*.test')))
        cv_train_data = sorted(map(lambda p: str(p), cv_path.glob('set_*/*.train')))
        cv_valid_data = sorted(map(lambda p: str(p), cv_path.glob('set_*/*.train')))
        cv_sets = list(zip(cv_train_data, cv_valid_data))
    
    # Ensure it's an int.
    THREADS = int(THREADS)

    print("I am using {} threads".format(THREADS))
    params_csv = pd.read_csv(parameters_file_name)
    params_csv['f1_cross_validation'] = None
    params_csv['f1_test'] = None
    
    eval_model_parameters(params_csv, cv_sets)
    eval_test_data(params_csv, train_file_name, test_file_name)

    params_csv.to_csv(result_csv, index=False)


if __name__ == "__main__":
    main()
