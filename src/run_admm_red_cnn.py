import dival
import matplotlib.pyplot as plt

# TODO: set dival config location to /w/383/johnlin/.dival/
# TODO: set dival lodopab_dataset to /w/331/yukthiw/tmp/lodopab
# Ref: https://jleuschn.github.io/docs.dival/dival.config.html
def main():
    dataset = dival.datasets.get_standard_dataset("lodopab")
    print(dataset)


if __name__ == "__main__":
    main()
