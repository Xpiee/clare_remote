#!/bin/bash -l

#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch,Combined


SBATCH --output=logs/err/%j.out
SBATCH --error=logs/err/%j.err

python -u t_3.py >logs/t_3.py.log 2>&1