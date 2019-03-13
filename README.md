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

## Visualize the workflow
Snakemake can visualize the workflow using ```dot```. Run the following to generate a png for the workflow.
```bash
$ snakemake --dag sort_f1_scores | dot -Tpng > dag.png
```
