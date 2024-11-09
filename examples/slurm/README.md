# Slurm Python Example

This example demonstrates how to run a Python script using Slurm workload manager.
The example uses only Python standard library modules and requires no additional dependencies.

## Project Structure

```
examples/slurm/
├── run_job.sh          # Slurm job submission script
├── data/
│   ├── input.txt       # Example input file
│   └── output.txt      # Example output file
├── src/
│   ├── __init__.py
│   └── process_data.py # Main processing script
└── README.md          # This file
```

## Usage

1. Prepare your input data file
2. Submit the job to Slurm:

```bash
sbatch run_job.sh input_file.txt output_file.txt
```

## Features

- Parallel processing using Python's multiprocessing
- Logging to both file and console
- Command-line argument parsing
- Error handling and reporting
- No external dependencies - uses only Python standard library

## Monitoring

Check job status:

```bash
squeue -u $USER
```

Check job logs:

```bash
cat logs/<job_id>.log
```
