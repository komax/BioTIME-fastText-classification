# Classification using fastText

This respository uses biodiversity data from the [BioTIME database](http://biotime.st-andrews.ac.uk/downloadFull.php) to classifiy methods texts using [fastText](https://fasttext.cc/).

## Requirements

- a local copy of BioTIME and the metadata.
- conda (Miniconda or Anaconda)
- a Python fastText binding (more information in the installation section)

## Install
This sections guides you to set up this project to run experiments using Snakemake and fastText.


### 1. Clone the respository

```bash
$ git clone https://github.com/komax/BioTIME-fastText-classification
```

### 2. Anaconda environment
1. Create a new environment, e.g., ```biotime-fasttext``` and install all dependencies.
```bash
$ conda env create --name biotime-fasttext --file environment.yaml
```

2. Activate the conda environment. Either use the anaconda navigator or use this command in your terminal:
```bash
$ conda activate biotime-fasttext
```
or
```bash
$ source activate biotime-fasttext
```

### 3. Python bindings for fastText
Disclaimer: you can use ```pip install fasttext``` in your anaconda environment, but those bindings are outdated.


I recommend doing this:
0. First activate your anaconda environment.
1. Checkout the github respository from fastText or a stable fork:
```bash
$ git clone https://github.com/komax/fastText
```
2. Install the python bindings in the fastText respository
```bash
pip install .
```

### 4. Link or copy your BioTIME data to the respository
Create a symlink or copy your BioTIME data into ```biotime``` directory.

### 5. Ensure to download ```punkt``` from nltk
nltk requires to download content to tokenize a sentence. Run this in your python shell:
```python
>>> import nltk
>>> nltk.download('punkt')
```
or run
```bash
$ python scripts/download-nltk-punkt.py
```



## Run experiments with Snakemake
All configuration parameters are stored in ```Snakefile```. Change the parameters to your purpose.
Adjust ```-j <num_cores>``` in your snakemake calls to make use of multiple cores to run at the same time.

### 1. Data preparation
```bash
$ snakemake normalize_fasttext
```

### 2. Cross validation
Create data for cross validation, split the model parameters up in blocks and sort the model parameters by f1 scores on the training data.
```bash
$ snakemake sort_f1_scores
```

### 3. Train a model
Select the best model (from the cross validation) and train it
```bash
$ snakemake train_model
```

### 4. Testing a model
```bash
$ snakemake test_model
```

### 5. Run the entire pipeline
```bash
$ snakemake
```

## Visualize the workflow
Snakemake can visualize the workflow using ```dot```. Run the following to generate a png for the workflow.
```bash
$ snakemake --dag all | dot -Tpng > dag.png
```

## Customize the experiments
Checkout the ```Snakefile``` and adjust this section to configure the experimental setup (parameter selection, cross validation, parallelization):
```python
KFOLD = 2
TEST_SIZE = 0.25
CHUNKS = 4
PARAMETER_SPACE = ModelParams(
    dim=ParamRange(start=10, stop=100, num=2),
    lr=ParamRange(start=0.1, stop=1.0, num=2),
    wordNgrams=ParamRange(start=2, stop=5, num=2),
    epoch=ParamRange(start=5, stop=50, num=2),
    bucket=ParamRange(start=2_000_000, stop=10_000_000, num=2)
)
FIRST_N_SENTENCES = 1
```

## Inspect the experimental results
The (sub)directory ```data``` contains intermediate data from data transforms/selection, chunking of the parameter space ```data/blocks``` and subsampling for cross validation ```data/cv```.

```results``` entails the parameterization for the experiments as well as the accurancy scores measured as [f1 scores](https://en.wikipedia.org/wiki/F1_score) on precision and recall:
 * ```results/blocks``` contains all chunks (inlcuding the validation scores) as ```csv```s,
 * ```results/params_scores.csv``` is the concatenation of all blocks,
 * ```results/params_scores_sorted.csv``` ranks the resulting scores by the ```f1_cross_validation_micro``` score on the cross validation sets per label. Then, we select the model with the smallest ```f1_cross_validation_micro_ptp``` with smallest point to point distance (minimum value to maximium value)
