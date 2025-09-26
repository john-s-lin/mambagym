import argparse
import io
import logging
import os
import zipfile
from pathlib import Path

import requests

DATASET_FULL = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/ct-low-dose-reconstruction"

log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper()))


def unzip(content: io.BytesIO, target_dir: Path | str) -> None:
    try:
        logging.info(f"Extracting ZIP file to '{target_dir}/'")
        with zipfile.ZipFile(content, "r") as z:
            z.extractall(path=target_dir)
    except zipfile.BadZipFile as e:
        logging.error(f"Error processing ZIP file: {e}")


def main():
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
