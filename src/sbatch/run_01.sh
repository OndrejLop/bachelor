#!/bin/bash
#SBATCH --partition=gpu-bio             # partition you want to run job in
#SBATCH --account=ksibio  
#SBATCH --gres=gpu:1                    # request 1 GPU
#SBATCH --job-name=extract_seq
#SBATCH --output=extract_seq.log
#SBATCH --time=01:00:00
#SBATCH --mem=4G

cd /home/lopatkao/bachelor/git/
source /home/lopatkao/esm-env312/bin/activate
python3 src/scripts/01_extract_sequence.py
