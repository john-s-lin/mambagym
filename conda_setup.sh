#!/usr/bin/env bash

#SBATCH --job-name=conda_setup
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1

TARGET_DIR_PREFIX="/w/383/${USER}"

# Make a logs directory if it doesn't exist
mkdir -p logs

# Create an alternate path for conda environments and conda packages and 
# add them to .condarc configuration
if [ ! -d "${TARGET_DIR_PREFIX}/conda_envs" ]; then
    mkdir -p "${TARGET_DIR_PREFIX}/conda_envs"
    conda config --add envs_dirs "${TARGET_DIR_PREFIX}/conda_envs"
else
    echo "${TARGET_DIR_PREFIX}/conda_envs already exists"
fi

if [ ! -d "${TARGET_DIR_PREFIX}/conda_pkgs" ]; then
    mkdir -p "${TARGET_DIR_PREFIX}/conda_pkgs"
    conda config --add pkgs_dirs "${TARGET_DIR_PREFIX}/conda_pkgs"
else
    echo "${TARGET_DIR_PREFIX}/conda_pkgs already exists"
fi

# Only create a new environment if it doesn't exist
if conda info --envs | grep -q mambagym; then
    echo "mambagym already exists"
else
    conda env create -f environment.yml -p "${TARGET_DIR_PREFIX}/${USER}/conda_envs/mambagym" -v -y
fi

exit 0