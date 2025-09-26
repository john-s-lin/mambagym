import argparse
import io
import logging
import multiprocessing as mp
import os
import zipfile
from functools import partial
from pathlib import Path

import requests

DATASET_FULL = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/ct-low-dose-reconstruction"

log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper()))


def extract_single(z: zipfile.ZipFile, target_dir: Path | str, file: str) -> None:
    """Extracts a single file from a ZIP archive to the target directory.

    Args:
        z (zipfile.ZipFile): The ZIP file object to extract from.
        target_dir (Path | str): The directory path to extract the file to.
        file (str): The name of the file within the ZIP archive to extract.
    """
    z.extract(file, path=target_dir)


def unzip(content: io.BytesIO, target_dir: Path | str) -> None:
    """Extracts a ZIP file from BytesIO to the target directory.

    Args:
        content (io.BytesIO): The ZIP file content as a BytesIO object.
        target_dir (Path | str): The directory path to extract the ZIP contents to.
    """
    try:
        logging.info(f"Extracting ZIP file to '{target_dir}/'")
        with zipfile.ZipFile(content, "r") as z:
            file_list = z.namelist()
            total_files = len(file_list)
            logging.info(f"Total files to extract: {total_files}")

            num_processes = min(mp.cpu_count(), total_files)
            with mp.Pool(num_processes) as pool:
                extract_func = partial(extract_single, z, target_dir)
                pool.map(extract_func, file_list)
            logging.info("Extraction completed.")
    except zipfile.BadZipFile as e:
        logging.error(f"Error processing ZIP file: {e}")


def main():
    """Downloads training and test data from the specified URL to the output directory.

    This function parses command-line arguments for input URL and output directory,
    downloads the dataset, and extracts it to the specified location.
    """
    parser = argparse.ArgumentParser(description="Downloads training and test data")
    parser.add_argument("-i", "--input", default=DATASET_FULL, type=str)
    parser.add_argument("-o", "--output", default="data", type=str)
    args = parser.parse_args()

    input_url = args.input
    output_path = args.output

    try:
        os.makedirs(output_path, exist_ok=True)
        logging.debug(f"Made output data directory at '{output_path}'")

        logging.info(f"Making GET request to '{input_url}'")
        response: requests.Response = requests.get(input_url)
        response.raise_for_status()

        unzip(io.BytesIO(response.content), target_dir=output_path)
    except requests.RequestException as e:
        logging.error(f"Error downloading file from '{input_url}': {e}")
    except Exception as e:
        logging.error(f"An unexpected error occured: {e}")


if __name__ == "__main__":
    main()
