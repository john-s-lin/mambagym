import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add root directory to Python path
# TODO this is a hack, don't do this
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Helper code located in DenoMamba repo
# TODO move to utils folder? This will cause Ruff to complain
from DenoMamba.data import create_loaders_mix
from DenoMamba.options import TrainOptions
from models.red_cnn.solver import Solver

FIG_PATH = Path(__file__).parents[1].joinpath("out/figures").resolve()


def get_latest_checkpoint(checkpoint_dir: str) -> int:
    pattern = r"REDCNN_(\d+)_epoch.ckpt"
    latest_epoch = 0

    for filename in os.listdir(checkpoint_dir):
        match_name = re.match(pattern, filename)
        if match_name:
            epoch_num = int(match_name.group(1))
            latest_epoch = max(latest_epoch, epoch_num)

    if latest_epoch == 0:
        print("Checkpoints directory is empty. No iterations found.")

    return latest_epoch


def train(dataloader: DataLoader, checkpoint_dir: str):
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
        save_path=checkpoint_dir,
        save_fig_path=FIG_PATH,
        multi_gpu=False,
        num_epochs=100,
        print_iters=20,
        decay_iters=3000,
        save_iters=1000,
        test_epochs=99,
        result_fig=True,
        patch_size=64,
        lr=1e-5,
    )

    red_cnn_solver = Solver(solver_args, dataloader)

    latest_epoch = get_latest_checkpoint(checkpoint_dir=solver_args.save_path)

    if latest_epoch >= solver_args.num_epochs - 1:
        print(f"Found existing model at epoch {latest_epoch}. Loading checkpoint...")
        red_cnn_solver.load_model(latest_epoch)
    else:
        if latest_epoch > 0:
            print(f"Found partial training checkpoint at epoch {latest_epoch}. Resuming training...")
            red_cnn_solver.load_model(latest_epoch)
            red_cnn_solver.train(last_epoch=latest_epoch)
        else:
            print("No existing checkpoints found. Starting training from epoch 0...")
            red_cnn_solver.train()


def validate(dataloader: DataLoader, checkpoint_dir: str):
    solver_args = argparse.Namespace(
        mode="test",
        load_mode=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        norm_range_min=-1024.0,
        norm_range_max=3072.0,
        trunc_min=-160.0,
        trunc_max=240.0,
        save_path=checkpoint_dir,
        save_fig_path=FIG_PATH,
        multi_gpu=False,
        num_epochs=100,
        print_iters=20,
        decay_iters=3000,
        save_iters=1000,
        test_epochs=99,
        result_fig=True,
        patch_size=64,
        lr=1e-5,
    )

    red_cnn_solver = Solver(solver_args, dataloader)

    latest_epoch = get_latest_checkpoint(solver_args.save_path)

    if latest_epoch >= solver_args.num_epochs - 1:
        print("Running validation...")
        red_cnn_solver.test()
    else:
        print("Solver is not fully trained, try training again")


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

    train(dataloader=trainloader, checkpoint_dir=path_to_save)
    validate(dataloader=validloader, checkpoint_dir=path_to_save)


if __name__ == "__main__":
    main()
