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


def write_study_line_number_mapping(metadata_dataframe, filename):
    study_ids = metadata_dataframe["STUDY_ID"]
    study_ids.to_csv(filename, index=True, header=['STUDY_ID'], index_label='LINENUMBER')
    return


def write_metadata_fasttext(metadata_df, outfilename, firstSentences=3):
    # Select first n sentences, then trim them.
    metadata_df["METHODS"] = metadata_df["METHODS"].map(firstNsentences).map(trim_sentence)
    metadata_df.pop('STUDY_ID')
    metadata_df.to_csv(outfilename, index=False, quoting=csv.QUOTE_ALL)


biotime_df = pd.read_csv(snakemake.input.csv, encoding="ISO-8859-1")
write_study_line_number_mapping(biotime_df, snakemake.output.map)
write_metadata_fasttext(biotime_df, snakemake.output.txt, firstSentences=3)
