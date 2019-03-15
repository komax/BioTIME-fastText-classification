#!/bin/bash

#$ -S /bin/bash
#$ -wd /work/konzack/biotime-fasttext

#$ -l h_rt=24:00:00
#$ -l h_vmem=8G
#$ -pe smp 2-28

#$ -j y
#$ -o /work/$USER/$JOB_NAME-$JOB_ID.out

if [[ ! $NSLOTS ]]; then
    NSLOTS=`getconf _NPROCESSORS_ONLN`
fi

module load anaconda
source activate biotime-fasttext
python scripts/cross-validate-classifier.py $1 $2 $3 $NSLOTS
