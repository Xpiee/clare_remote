#!/bin/bash -l

#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch,Combined


# SBATCH --output=logs/err/%j.out
SBATCH --error=logs/err/%j.err

python -u inference_time_calc.py >logs/inference_time_calc.py.log 2>&1