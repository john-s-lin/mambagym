import argparse


def main():
    parser = argparse.ArgumentParser(description="Downloads training and test data")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["full", "small"],
        default="full",
        help="Dataset size (default: full)",
    )
    parser.add_argument("--output", "-o")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
