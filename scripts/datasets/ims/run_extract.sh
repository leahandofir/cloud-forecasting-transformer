#!/bin/bash

#SBATCH --job-name=extract_fetch
#SBATCH --error=%x-%j.out
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00

module load anaconda3

source activate leahandofir

/ims_home/orenp/.conda/envs/leahandofir/bin/python -u extract_ims_data.py > extract_prints_%j.txt


