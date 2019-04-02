#!/usr/bin/env python

import re
import csv

from nltk import tokenize
import pandas as pd


def prune_urls(text):
    return re.sub(r'(https?://[^\s]+)', ' ', text)


def trim_sentence(sentence):
    return prune_urls(sentence.replace('\n', ' ').replace('\r', ' ').replace('&amp;', '&'))


def firstNsentences(text, n=3):
    sentences = tokenize.sent_tokenize(text)
    n_sentences = sentences[:n]
    return " ".join(n_sentences)


def remove_studies_wtih_empty_classes(dataframe):
    column_name = label_column_name(dataframe)
    return dataframe[pd.notnull(dataframe[column_name])]


def write_study_line_number_mapping(metadata_dataframe, filename):
    study_ids = metadata_dataframe["STUDY_ID"]
    study_ids.to_csv(filename, index=True, header=['STUDY_ID'], index_label='LINENUMBER')
    return


def label_column_name(df):
    return df.columns[snakemake.params.label_column]


def replace_class_name_by_num(data_frame):
    column_name = label_column_name(data_frame)
    
    label_values = data_frame[column_name].unique().tolist()
    remapping = dict()

    for i, label in enumerate(label_values, start=1):
        remapping[label] = i

    data_frame[column_name] = data_frame[column_name].map(remapping)
    return data_frame


def write_metadata_fasttext(metadata_df, outfilename, firstSentences=3):
    # Select first n sentences, then trim them.
    metadata_df["METHODS"] = metadata_df["METHODS"].map(firstNsentences).map(trim_sentence)
    metadata_df.pop('STUDY_ID')
    # Write the data as csv (no header).
    metadata_df.to_csv(outfilename, header=False, index=False, quoting=csv.QUOTE_ALL)


biotime_df = pd.read_csv(snakemake.input.csv, encoding="ISO-8859-1")
# Trim data.
biotime_df = remove_studies_wtih_empty_classes(biotime_df)
if snakemake.params.replace_labels_by_num:
    biotime_df = replace_class_name_by_num(biotime_df)

# Write the data.
write_study_line_number_mapping(biotime_df, snakemake.output.map)
write_metadata_fasttext(biotime_df, snakemake.output.txt, firstSentences=snakemake.params.firstNSentences)
