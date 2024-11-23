from datetime import datetime

import dival
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset


# TODO: set dival config location to /w/383/johnlin/.dival/
# Ref: https://jleuschn.github.io/docs.dival/dival.config.html
def main():
    dataset = dival.datasets.get_standard_dataset("lodopab")
    data = dataset.create_torch_dataset("train")
    data = Subset(data, indices=range(200))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
