#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=v100
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out

module load conda/latest
conda activate csc592-llms

cd /work/pi_csc592_uri_edu/calvin/csc592-llms/hw-07

uv run train.py
