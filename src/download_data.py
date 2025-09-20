import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Downloads training and test data"
    )
    parser.add_argument("--input", "-i", required=True)
    parser.parse_args()
    print(parser)

if __name__ == "__main__":
    main()
