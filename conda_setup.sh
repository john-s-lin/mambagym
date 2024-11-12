#!/usr/bin/env bash

#SBATCH --job-name=conda_setup
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1

mkdir -p logs

conda env create -f environment.yml -p "/w/383/${USER}/conda_envs/mambagym" -v -y

exit 0