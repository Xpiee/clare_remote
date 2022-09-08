#!/bin/bash -l

#SBATCH --account=ingenuitylabs
#SBATCH --partition=Sasquatch,Combined


SBATCH --output=logs/err/%j.out
SBATCH --error=logs/err/%j.err

python -u t_ecg_eda_eeg.py >logs/t_ecg_eda_eeg.py.log 2>&1