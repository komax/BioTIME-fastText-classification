
def foo():
    pass

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
        "data/biotime_metadata_prep.csv"
    output:
        "data/biotime_prep_fasttext.txt",
        "data/biotime_linenumber_studyid_mapping.csv"
    script:
        "scripts/data-prep-fasttext.py"

rule normalize_fastext:
    input:
        "data/biotime_prep_fasttext.txt"
    output:
        "data/biotime_fasttext.txt"
    script:
        "scripts/normalize-fasttext.sh {input} > {output}"
