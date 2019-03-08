
import math

import numpy as np
import fastText

KFOLD = 3
CV_EXTS = ['train','valid']

rule gen_parameters:
    output:
        "data/params.csv"
    script:
        "scripts/generate-parameter-space.py"

rule select_columns_biotime:
    input:
        "biotime/BioTIMEMetadata_02_04_2018.csv"
    output:
        "data/biotime_metadata_prep.csv"
    shell:
        "xsv select STUDY_ID,REALM,TAXA,METHODS {input} > {output}"

rule data_prep_fasttext:
    input:
        csv="data/biotime_metadata_prep.csv"
    params:
        firstNSentences=3
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

rule train_model:
    input:
        "data/biotime_fasttext.txt"
    output:
        mod="models/biotime_metadata_model.bin"
    run:
        import os
        model = os.path.splitext(output.mod)[0]
        shell("fasttext supervised -input {input} -output {model} -dim 10 -lr 1.5 -wordNgrams 3 -minCount 1 -bucket 10000000 -epoch 50 -thread 4")

rule split_data:
    input:
        data="data/biotime_fasttext.txt"
    params:
        kfold=KFOLD
    output:
        test_data="data/cv/biotime.test",
        train_data="data/cv/biotime.train",
        cv_data = expand("data/cv/set_{cv_subset}/biotime.{cv_ext}", cv_subset=range(KFOLD), cv_ext=CV_EXTS)
    script:
        "scripts/split-data.py"

rule cross_validate:
    input:
        train_data=expand("data/cv/set_{cv_set}/biotime.train", cv_set=range(KFOLD)),
        valid_data=expand("data/cv/set_{cv_set}/biotime.valid", cv_set=range(KFOLD))
    params:
        kfold=KFOLD
    script:
        "scripts/cross-validate-classifier.py"

rule find_parameters:
    input:
        train_data=expand("data/cv/set_{cv_set}/biotime.train", cv_set=range(KFOLD)),
        valid_data=expand("data/cv/set_{cv_set}/biotime.valid", cv_set=range(KFOLD)),
        test_data="data/cv/biotime.test",
        train_file="data/cv/biotime.train"
    run:
        best_score = -math.inf
        cv_sets = list(zip(input.train_data, input.valid_data))
        best_dim, best_lr, best_wordNgrams, best_epoch, best_bucket = None, None, None, None, None
        for dim, lr, wordNgrams, epoch, bucket in parameter_combinations():
            f1_scores = []
            for train_file, valid_file in cv_sets:
                print(dim, lr, wordNgrams, epoch, bucket)
                classifier = fastText.train_supervised(train_file,
                    dim=dim, lr=lr, wordNgrams=wordNgrams, minCount=1, bucket=bucket, epoch=epoch, thread=4)
                _, precision, recall = classifier.test(valid_file)
                f1_scores.append(f1_score(precision, recall))
            mean_f1 = sum(f1_scores) / KFOLD
            if mean_f1 > best_score:
                print("found a better one")
                best_score = mean_f1
                best_dim, best_lr, best_wordNgrams, best_epoch, best_bucket = dim, lr, wordNgrams, epoch, bucket

        print("Optimal parameters:")
        print(best_dim, best_lr, best_wordNgrams, best_epoch, best_bucket)
        print(f"f1_score on training data={best_score}")
        classifier = fastText.train_supervised(input.train_file,
            dim=best_dim, lr=best_lr, wordNgrams=best_wordNgrams, minCount=1, bucket=best_bucket, epoch=best_epoch, thread=4)

        _, precision, recall = classifier.test(input.test_data)
        f1_test = f1_score(precision, recall)
        print(f"f1_score on test data={f1_test}")

rule clean:
    input:
        data=directory("data"),
        models=directory("models")
    run:
        shell("test -d {input.data} && rm -r {input.data}")
        shell("test -d {input.models} && rm -r {input.models}")
