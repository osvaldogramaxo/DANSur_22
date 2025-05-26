#!/bin/bash
#SBATCH --job-name=NNSur_sxs
#SBATCH --partition=normal-a100-40
#SBATCH --account=f202414020cpcaa3g
#SBATCH --nodes=1

#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH -o logs/train/sxs_out.log
#SBATCH -e logs/train/sxs_error.log

source ~/.bashrc

ENVIRONMENT="torch"
eval "$(conda shell.bash hook)"
conda activate $ENVIRONMENT

echo ">>>> NVIDIA INFO"
set -x #echo on
nvidia-smi
# set +x #echo off
# sleep 10
python3 ./sxs_finetune.py