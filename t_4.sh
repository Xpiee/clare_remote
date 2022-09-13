#!/bin/bash -l

#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch,Combined


# SBATCH --output=logs/err/%j.out
SBATCH --error=logs/err/%j.err

python -u training_four.py >logs/training_four1.py.log 2>&1