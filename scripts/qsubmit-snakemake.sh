#!/bin/bash

#$ -S /bin/bash
#$ -wd /work/konzack/biotime-fasttext

#$ -N biotime-fasttext-snakemake-small

#$ -l h_rt=24:00:00
#$ -l h_vmem=12G
#$ -pe smp 2-28

#$ -j y
#$ -o /work/$USER/$JOB_NAME-$JOB_ID.out

if [[ ! $NSLOTS ]]; then
    NSLOTS=`getconf _NPROCESSORS_ONLN`
fi

module load anaconda
source activate biotime-fasttext
snakemake sort_f1_scores -j $NSLOTS