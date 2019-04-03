#! /usr/bin/env python

import pandas as pd


parameter_scores = pd.read_csv(snakemake.input.params_scores)

sorted_scores = parameter_scores.sort_values(by=['f1_cross_validation_micro','f1_cross_validation_micro_ptp'],
ascending=[False,True] )

sorted_scores.to_csv(snakemake.output.scores_sorted, index=False)