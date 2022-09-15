#!/bin/bash -l

#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch,Combined


# SBATCH --output=logs/err/%j.out
# SBATCH --error=logs/err/%j.err

python -u training_unimodal_10fold.py >logs/training_unimodal_10fold.py.log 2>&1