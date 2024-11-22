import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from tqdm import tqdm

# Add root directory to Python path
# TODO this is a hack, don't do this
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Helper code located in DenoMamba repo
# TODO move to utils folder? This will cause Ruff to complain
from DenoMamba.data import create_loaders_mix
from DenoMamba.options import TrainOptions
from models.red_cnn.networks import RED_CNN
from models.red_cnn.solver import Solver


def get_pixel_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(target, prediction)


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float | int:
    mse = torch.nn.functional.mse_loss(original, reconstructed)
    if mse == 0:
        return float("inf")

    max_pixel_value = 1

    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    train_options = TrainOptions()
    args = train_options.parse()

    full_dose_path = args.full_dose_path
    quarter_dose_path = args.quarter_dose_path
    dataset_ratio = args.dataset_ratio
    train_ratio = args.train_ratio
    batch_size = args.batch_size
    path_to_save = args.path_to_save

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    trainloader, validloader = create_loaders_mix(
        full_dose_path=full_dose_path,
        quarter_dose_path=quarter_dose_path,
        dataset_ratio=dataset_ratio,
        train_ratio=train_ratio,
        batch_size=batch_size,
    )

    # This is a hack to pass in arguments manually to the RED_CNN trainer
    # From models/red_cnn/main.py, using the defaults
    solver_args = argparse.Namespace(
        mode="train",
        load_mode=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        norm_range_min=-1024.0,
        norm_range_max=3072.0,
        trunc_min=-160.0,
        trunc_max=240.0,
        save_path=path_to_save,
        multi_gpu=False,
        num_epochs=100,
        print_iters=20,
        decay_iters=3000,
        save_iters=1000,
        test_iters=1000,
        result_fig=True,
        patch_size=64,
        lr=1e-5,
    )

    red_cnn_solver = Solver(solver_args, trainloader)
    red_cnn_solver.train()


if __name__ == "__main__":
    main()
