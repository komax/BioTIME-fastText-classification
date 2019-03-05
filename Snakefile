
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
