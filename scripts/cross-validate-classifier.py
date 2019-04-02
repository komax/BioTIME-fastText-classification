#! /usr/bin/env python

import numpy as np
import pandas as pd
import fastText

from utils import f1_score, ResultScore

THREADS = 1


def scores_per_label(model, path_test_file, k=1, threshold=0.0):
    scores = model.test_label(path_test_file, k=k, threshold=threshold)

    result = pd.DataFrame(columns=['label', 'precision', 'recall', 'f1score'])

    for label, score in scores.items():
        result = result.append({'label':label, 'precision':score['precision'], 'recall':score['recall'], 'f1score':score['f1score']}, ignore_index=True)
    return result


def avg_result_score(result_score):
    result = ResultScore(precision=np.mean(result_score.precision),
        recall=np.mean(result_score.recall), f1score=np.mean(result_score.f1score))
    return result

# FIXME remove this.
def peak_to_peak_result_score(result_score):
    ResultScore(precision=np.ptp(result_score.precision),
        recall=np.ptp(result_score.recall),
        f1score=np.ptp(result_score.f1score))


def min_result_score(result_score):
    return ResultScore(precision=np.min(result_score.precision),
        recall=np.min(result_score.recall), f1score=np.min(result_score.f1score))


def max_result_score(result_score):
    return ResultScore(precision=np.max(result_score.precision),
        recall=np.max(result_score.recall), f1score=np.max(result_score.f1score))


def cross_validate_model(model_parameters, cross_validation_sets):
    macro_results = ResultScore(precision=[], recall=[], f1score=[])
    
    micro_result = pd.DataFrame(columns=['cross_valid', 'label', 'precision', 'recall', 'f1score'])

    for cv_id, (train_file, valid_file) in enumerate(cross_validation_sets):
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

        # Validation per class.
        score_for_cv_set = scores_per_label(classifier, valid_file)
        number_entries = len(score_for_cv_set.index)
        # Enter micro scores to results table across cross validation sets.
        repeated_cv_name_df = pd.DataFrame(number_entries * [f"cv{cv_id}"], columns=["cross_valid"])
        reshaped_scores = pd.concat([repeated_cv_name_df, score_for_cv_set], axis=1)
        micro_result = micro_result.append(reshaped_scores)

    # Reduce step: Calculate the means from the lists.
    macro_result = avg_result_score(macro_results)

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
    # TODO Store f1 from micro on this test set.
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

        params_csv.at[params.Index, 'precision_micro'] = results['micro'].precision.mean()
        params_csv.at[params.Index, 'recall_micro'] = results['micro'].recall.mean()
        params_csv.at[params.Index, 'f1_cross_validation_micro'] = results['micro'].f1score.mean()

        params_csv.at[params.Index, 'precision_micro_ptp'] = results['micro'].precision.ptp()
        params_csv.at[params.Index, 'recall_micro_ptp'] = results['micro'].recall.ptp()
        params_csv.at[params.Index, 'f1_cross_validation_micro_ptp'] = results['micro'].f1score.ptp()


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

    params_csv['precision_micro_ptp'] = None
    params_csv['recall_micro_ptp'] = None
    params_csv['f1_cross_validation_micro_ptp'] = None

    params_csv['f1_test'] = None
    
    eval_model_parameters(params_csv, cv_sets)
    eval_test_data(params_csv, train_file_name, test_file_name)

    params_csv.to_csv(result_csv, index=False)
    print("I completed my run. bye")


if __name__ == "__main__":
    main()
