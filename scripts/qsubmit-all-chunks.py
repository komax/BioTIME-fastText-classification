#!/usr/bin/env python

import sys

from pathlib import Path
from subprocess import call


def submit_job(params_chunk, cv_dir, results_dir):
    #func_call = ["./scripts/qsubmit-params-chunk.sh"]
    func_call = ['qsub',
        '-N', f'cross-validate-{params_chunk.stem}',
        "./scripts/qsubmit-params-chunk.sh"]
    out_csv = Path(results_dir) / params_chunk.name.replace("params", "scores")
    print(params_chunk.name.replace("params", "scores"))
    func_call.extend([
        str(params_chunk), cv_dir, str(out_csv)
    ])
    print(" ".join(func_call))
    #call(func_call)


def cross_validate_all_chunks(params_blocks_dir, cv_dir, results_dir):
    for chunk_csv in params_blocks_dir.glob('params_*.csv'):
        submit_job(chunk_csv, cv_dir, results_dir)



def main():
    _, params_chunks_dir, cv_dir, results_dir = sys.argv
    params_chunks_path = Path(params_chunks_dir)
    Path(cv_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    cross_validate_all_chunks(params_chunks_path, cv_dir, results_dir)


if __name__ == "__main__":
    main()