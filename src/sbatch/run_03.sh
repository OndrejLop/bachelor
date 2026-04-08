#!/bin/bash
#SBATCH --partition=gpu-bio             # partition you want to run job in
#SBATCH --account=ksibio  
#SBATCH --gres=gpu:1                    # request 1 GPU
#SBATCH --time=20:00:00                 # walltime for the job in format (days-)hours:minutes:seconds
#SBATCH --nodes=1                       # number of nodes (can be only 1)
#SBATCH --mem=512000                    # memory resource per node
#SBATCH --job-name="crypto_show"        # change to your job name
#SBATCH --output=output.txt             # stdout and stderr output file
#SBATCH --mail-user=lopatkao@natur.cuni.cz # send email when job changes state to email address user@example.com
#SBATCH --exclusive                     # Use whole node

cd /home/lopatkao/bachelor/git/
source /home/lopatkao/esm-env312/bin/activate
python3 src/scripts/03_cluster_pockets.py #--decision-threshold 0.7 --distance-threshold 10