#!/usr/bin/env python

import pandas as pd
import numpy as np

def parameter_combinations():
    for dim in np.linspace(start=10, stop=1000, num=100, dtype=int):
        for lr in np.linspace(start=0.1, stop=3.0, num=30):
            for wordNgrams in np.linspace(start=2, stop=5, num=4, dtype=int):
                for epoch in np.linspace(start=5, stop=50, num=8, dtype=int):
                    for bucket in np.linspace(start=2_000_000, stop=10_000_000, num=5, dtype=int):
                        yield dim, lr, wordNgrams, epoch, bucket

df = pd.DataFrame(parameter_combinations(), columns=['dim', 'lr', 'wordNgrams', 'epoch', 'bucket'])
df.to_csv(snakemake.output[0], index=False)
