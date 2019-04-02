#! /usr/bin/env python

import numpy as np
import pandas as pd
import fastText

from utils import f1_score, ResultScore

THREADS = 1


def scores_pooled_from_labels(model, path_test_file, k=1, threshold=0.0):
    scores = model.test_label(path_test_file, k=k, threshold=threshold)
    precisions = list()
    recalls = list()
    f1_scores = list()

    for score in scores.values():
        precisions.append(score['precision'])
        recalls.append(score['recall'])
        f1_scores.append(score['f1score'])

    pooled_precision = np.mean(precisions)
    pooled_recall = np.mean(recalls)
    pooled_f1 = np.mean(f1_scores)
    return ResultScore(pooled_precision, pooled_recall, pooled_f1)


def avg_result_score(result_score):
    result = ResultScore(precision=np.mean(result_score.precision),
        recall=np.mean(result_score.recall), f1score=np.mean(result_score.f1score))
    return result


def cross_validate_model(model_parameters, cross_validation_sets):
    micro_results = ResultScore(precision=[], recall=[], f1score=[])
    macro_results = ResultScore(precision=[], recall=[], f1score=[])
    for train_file, valid_file in cross_validation_sets:
        # Train the classifier.
        classifier = fastText.train_supervised(train_file,
            dim=model_parameters.dim, lr=model_parameters.lr,
            wordNgrams=model_parameters.wordNgrams, minCount=1,
            bucket=model_parameters.bucket, epoch=model_parameters.epoch, thread=THREADS)
        # Evaluate the classifier.
        # Test the model globally.
        _, precision, recall = classifier.test(valid_file)
        macro_results.precision.append(precision)
        macro_results.recall.append(recall)
        macro_results.f1score.append(f1_score(precision, recall))
        # Test the model on micro level (each class).
        result = scores_pooled_from_labels(classifier, valid_file)
        micro_results.precision.append(result.precision)
        micro_results.recall.append(result.recall)
        micro_results.f1score.append(result.f1score)

    # Reduce step: Calculate the means from the lists.
    macro_result = avg_result_score(macro_results)
    micro_result = avg_result_score(micro_results)
    results = {
        'macro': macro_result,
        'micro': micro_result
    }
    return results


def test_model(model_parameters, train_data_file, test_data_file):
    classifier = fastText.train_supervised(train_data_file,
            dim=model_parameters.dim, lr=model_parameters.lr,
            wordNgrams=model_parameters.wordNgrams, minCount=1,
            bucket=model_parameters.bucket, epoch=model_parameters.epoch, thread=THREADS)
    _, precision, recall = classifier.test(test_data_file)
    return f1_score(precision, recall)


def eval_model_parameters(params_csv, cv_sets):
    for params in params_csv.itertuples(index=True, name='ModelParams'):
        # Evaluate the model.
        results = cross_validate_model(params, cv_sets)

        # Store the results in the data frame.
        params_csv.at[params.Index, 'precision_macro'] = results['macro'].precision
        params_csv.at[params.Index, 'recall_macro'] = results['macro'].recall
        params_csv.at[params.Index, 'f1_cross_validation_macro'] = results['macro'].f1score

        params_csv.at[params.Index, 'precision_micro'] = results['micro'].precision
        params_csv.at[params.Index, 'recall_micro'] = results['micro'].recall
        params_csv.at[params.Index, 'f1_cross_validation_micro'] = results['micro'].f1score


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
        cv_valid_data = sorted(map(lambda p: str(p), cv_path.glob('set_*/*.valid')))
        cv_sets = list(zip(cv_train_data, cv_valid_data))
    
    # Ensure it's an int.
    THREADS = int(THREADS)

    print("Hi there, I am running with {} threads".format(THREADS))
    params_csv = pd.read_csv(parameters_file_name)
    params_csv['precision_macro'] = None
    params_csv['recall_macro'] = None
    params_csv['f1_cross_validation_macro'] = None
    
    params_csv['precision_micro'] = None
    params_csv['recall_micro'] = None
    params_csv['f1_cross_validation_micro'] = None

    params_csv['f1_test'] = None
    
    eval_model_parameters(params_csv, cv_sets)
    eval_test_data(params_csv, train_file_name, test_file_name)

    params_csv.to_csv(result_csv, index=False)
    print("I completed my run. bye")


if __name__ == "__main__":
    main()
