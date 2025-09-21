import argparse
import requests

DATASET_FULL = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/ct-low-dose-reconstruction"

def main():
    parser = argparse.ArgumentParser(description="Downloads training and test data")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
