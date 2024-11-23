#!/usr/bin/env bash

#SBATCH --job-name=train_red_cnn
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1

# Source conda initialization
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

FULL_DOSE_PATH="/w/331/yukthiw/tmp/temp_data/Preprocessed_256x256/256/Full Dose"
QUARTER_DOSE_PATH="/w/331/yukthiw/tmp/temp_data/Preprocessed_256x256/256/Quarter Dose"

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

python src/train_red_cnn.py --full_dose_path="${FULL_DOSE_PATH}" --quarter_dose_path="${QUARTER_DOSE_PATH}" \
    --path_to_save="./src/models/red_cnn/checkpoints"

exit 0
