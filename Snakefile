import pandas as pd

from scripts import utils
from scripts.utils import ModelParams
from scripts.utils import ParamRange

THREADS = 4

CV_EXTS = ['train','valid']

# Small example.
KFOLD = 2
CHUNKS = 4
PARAMETER_SPACE = ModelParams(
    dim=ParamRange(start=10, stop=100, num=2),
    lr=ParamRange(start=0.1, stop=1.0, num=2),
    wordNgrams=ParamRange(start=2, stop=5, num=2),
    epoch=ParamRange(start=5, stop=50, num=2),
    bucket=ParamRange(start=2_000_000, stop=10_000_000, num=2)
)
FIRST_N_SENTENCES = 3

# Comprehensive example.
# KFOLD = 10
# CHUNKS = 900
# PARAMETER_SPACE = ModelParams(
#    dim=ParamRange(start=10, stop=100, num=10),
#    lr=ParamRange(start=0.1, stop=1.0, num=10),
#    wordNgrams=ParamRange(start=2, stop=5, num=4),
#    epoch=ParamRange(start=5, stop=50, num=8),
#    bucket=ParamRange(start=2_000_000, stop=10_000_000, num=5)
#)
#FIRST_N_SENTENCES = 5000


rule all:
    input:
        model="models/biotime_metadata_model.bin"
    run:
        print("Done with training.")
        print(f"Best model is stored here: {input.model}")

rule gen_parameters:
    output:
        "data/params.csv"
    run:
        combinations = pd.DataFrame(utils.parameter_combinations(PARAMETER_SPACE), columns=['dim', 'lr', 'wordNgrams', 'epoch', 'bucket'])
        combinations.to_csv(output[0], index=False)

rule chunk_parameters:
    input:
        params_csv="data/params.csv"
    params:
        chunks=CHUNKS
    output:
        params_chunks_csv=expand("data/blocks/params_{chunk}.csv", chunk=range(CHUNKS))
    script:
        "scripts/chunk-parameters.py"

rule select_columns_biotime:
    input:
        "biotime/BioTIMEMetadata_02_04_2018.csv"
    output:
        "data/biotime_metadata_prep.csv"
    shell:
        #"xsv select STUDY_ID,REALM,TAXA,METHODS {input} > {output}"
        "xsv select STUDY_ID,REALM,METHODS {input} > {output}"

rule data_prep_fasttext:
    input:
        csv="data/biotime_metadata_prep.csv"
    params:
        firstNSentences=FIRST_N_SENTENCES
    output:
        txt="data/biotime_prep_fasttext.txt",
        map="data/biotime_linenumber_studyid_mapping.csv"
    script:
        "scripts/data-prep-fasttext.py"

rule normalize_fasttext:
    input:
        "data/biotime_prep_fasttext.txt"
    output:
        "data/biotime_fasttext.txt"
    shell:
        "source scripts/normalize-fasttext.sh &&"
        "normalize_text <{input} > {output}"

rule split_data:
    input:
        data="data/biotime_fasttext.txt"
    params:
        kfold=KFOLD,
        test_size=0.25
    output:
        test_data="data/cv/biotime.test",
        train_data="data/cv/biotime.train",
        cv_data = expand("data/cv/set_{cv_subset}/biotime.{cv_ext}", cv_subset=range(KFOLD), cv_ext=CV_EXTS)
    script:
        "scripts/split-data.py"

rule cross_validate_chunk:
    input:
        train_data=expand("data/cv/set_{cv_set}/biotime.train", cv_set=range(KFOLD)),
        valid_data=expand("data/cv/set_{cv_set}/biotime.valid", cv_set=range(KFOLD)),
        test_file="data/cv/biotime.test",
        train_file="data/cv/biotime.train",
        params_csv = "data/blocks/params_{chunk}.csv"
    threads: THREADS
    params:
        kfold=KFOLD
    output:
        result="results/blocks/scores_{chunk}.csv"
    script:
        "scripts/cross-validate-classifier.py"

rule merge_chunks:
    input:
        expand("results/blocks/scores_{chunk}.csv", chunk=range(CHUNKS))
    output:
        "results/params_scores.csv"
    shell:
        "xsv cat rows {input} > {output}"

rule sort_f1_scores:
    input:
        "results/params_scores.csv"
    output:
        "results/params_scores_sorted.csv"
    shell:
        "xsv sort --reverse --select f1_cross_validation_micro {input} > {output}"

rule train_model:
    input:
        train_data="data/cv/biotime.train",
        results="results/params_scores_sorted.csv"
    threads: THREADS
    output:
        model_bin="models/biotime_metadata_model.bin"
    run:
        model = os.path.splitext(output.model_bin)[0]
        results = pd.read_csv(input.results)
        params = results.iloc[0]
        shell("fasttext supervised -input {input.train_data} -output {model} \
        -dim {params.dim} -lr {params.lr} -wordNgrams {params.wordNgrams} \
        -minCount 1 -bucket {params.bucket} -epoch {params.epoch} -thread {threads}")

rule test_model:
    input:
        model="models/biotime_metadata_model.bin",
        test_data="data/cv/biotime.test"
    shell:
        "fasttext test {input.model} {input.test_data}"

rule clean:
    run:
        shell("rm -rf data models results")
