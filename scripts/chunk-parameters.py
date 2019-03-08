#!/usr/bin/env python

import numpy as np
import pandas as pd

params = pd.read_csv(snakemake.input.params_csv)
chunks = np.array_split(params, snakemake.params.chunks)

for chunk, filename in zip(chunks, snakemake.output.params_chunks_csv):
    chunk.to_csv(filename, index=False)

