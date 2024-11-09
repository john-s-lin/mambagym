import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool


def setup_logging(job_id):
    """Configure logging for the application."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/process_{job_id}.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def process_chunk(chunk):
    """Process a chunk of data."""
    # Simulate some CPU-intensive work
    time.sleep(0.1)
    # Example processing: convert lines to uppercase
    return [line.upper() for line in chunk]


def split_list(data, n):
    """Split a list into n roughly equal chunks."""
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Process data using Slurm")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output file"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of worker processes"
    )

    args = parser.parse_args()

    # Get Slurm job ID from environment, or use 'local' if not running in Slurm
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    logger = setup_logging(job_id)

    logger.info(f"Starting job {job_id} with {args.num_workers} workers")
    logger.info(f"Processing {args.input_file} -> {args.output_file}")

    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Read input file
        with open(args.input_file, "r") as f:
            data = f.readlines()

        # Split data into chunks for parallel processing
        chunks = split_list(data, args.num_workers)

        # Process chunks in parallel
        with Pool(args.num_workers) as pool:
            results = pool.map(process_chunk, chunks)

        # Flatten results and write to output file
        processed_data = [line for chunk in results for line in chunk]

        with open(args.output_file, "w") as f:
            f.writelines(processed_data)

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
