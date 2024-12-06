#!/usr/bin/env bash

#SBATCH --job-name=run_red_cnn
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

if conda env list | grep -q "^mambagym"; then
    # Check if the environment is already activated
    if [[ "${CONDA_DEFAULT_ENV}" != "mambagym" ]]; then
        # Activate the environment
        conda activate mambagym
        echo "Activating environment 'mambagym'" 
    else
        echo "Environment 'mambagym' is already activated"
    fi
else
    echo "Error: Environment 'mambagym' does not exist"
    echo "Available environments:"
    conda env list
fi

python src/run_fixed_pt_red_cnn.py

exit 0