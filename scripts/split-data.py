#! /usr/bin/env python

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from utils import load_data_as_array

def generate_cv_datasets(dataset, kFold=5):
    kf = KFold(n_splits=kFold, shuffle=True)
    return kf.split(dataset)


def split_train_test_data(data, test_size=0.25):
    return train_test_split(data, test_size=test_size, shuffle=True)

def write_data(data_array, filename):
    np.savetxt(filename, data_array, fmt='%s')


normalized_data = load_data_as_array(snakemake.input.data)
cv_data, test_data = split_train_test_data(normalized_data, test_size=snakemake.params.test_size)

# Write test data.
write_data(test_data, snakemake.output.test_data)
write_data(cv_data, snakemake.output.train_data)

for cv_set, (train_indices, validation_indices) in enumerate(generate_cv_datasets(cv_data, kFold=snakemake.params.kfold)):
    cv_path = f'{snakemake.params.cross_val_dir}/set_{cv_set}'
    write_data(cv_data[train_indices], f'{cv_path}/biotime.train')
    write_data(cv_data[validation_indices], f'{cv_path}/biotime.valid')
