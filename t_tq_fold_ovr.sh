#!/bin/bash -l

#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch,Combined


# SBATCH --output=logs/err/%j.out
# SBATCH --error=logs/err/%j.err

python -u train_trimodal_kfold_overlap.py >logs/train_trimodal_kfold_overlap.py.log 2>&1