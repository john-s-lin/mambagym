#!/bin/bash
#SBATCH --job-name=python_example    # Job name
#SBATCH --output=logs/%j.log         # Output file (%j expands to jobID)
#SBATCH --error=logs/%j.err          # Error file
#SBATCH --time=01:00:00              # Time limit hrs:min:sec
#SBATCH --nodes=1                    # Number of nodes requested
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=8GB                    # Memory limit

# Load Python module
module load python/3.9

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the Python script with parameters
python src/process_data.py \
    --input_file="${1:-data/input.txt}" \
    --output_file="${2:-data/output.txt}" \
    --num_workers=$SLURM_CPUS_PER_TASK