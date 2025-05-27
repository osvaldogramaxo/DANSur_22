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
    esac
done
ENVIRONMENT="torch"
eval "$(conda shell.bash hook)"
conda activate $ENVIRONMENT


python3 -m scripts.waveform_generation --sur $SUR
