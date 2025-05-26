#!/bin/bash
#SBATCH --job-name=generation_mmode
#SBATCH --account=f202414020cpcaa3x
#SBATCH --nodes=1
#SBATCH --partition=normal-x86

#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH -o logs/gen/out.log
#SBATCH -e logs/gen/error.log

source ~/.bashrc
# get sur argument
for i in "$@"; do
    case $i in
        --sur=*)
            SUR="${i#*=}"
            ;;
        --modes=*)
            MODES="${i#*=}"
            ;;
    esac
done
ENVIRONMENT="torch"
eval "$(conda shell.bash hook)"
conda activate $ENVIRONMENT

echo ">>>> NVIDIA INFO"
set -x #echo on
nvidia-smi
# set +x #echo off
# sleep 10
# MODES=$1

# python3 $HOME/scratch/ml_mock_data/ds3/script_train.py
python3 waveform_generation_multimode.py --modes $MODES --sur $SUR
