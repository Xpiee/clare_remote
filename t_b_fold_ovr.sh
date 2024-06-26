#!/bin/bash -l

#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch,Combined


# SBATCH --output=logs/err/%j.out
# SBATCH --error=logs/err/%j.err

python -u train_bimodal_kfold_overlap.py >logs/train_bimodal_kfold_overlap_mia1.py.log 2>&1