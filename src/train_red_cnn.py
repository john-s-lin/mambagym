import argparse
import os
import re
import sys

import numpy as np
import torch

# Add root directory to Python path
# TODO this is a hack, don't do this
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Helper code located in DenoMamba repo
# TODO move to utils folder? This will cause Ruff to complain
from DenoMamba.data import create_loaders_mix
from DenoMamba.options import TrainOptions
from models.red_cnn.solver import Solver


def get_latest_checkpoint(checkpoint_dir: str, min_iterations: int = None) -> tuple[bool, int | None, str | None]:
    pattern = r"REDCNN_(\d+)iter.ckpt"
    checkpoints = []

    if not os.path.exists(checkpoint_dir):
        return False, None, None

    for filename in os.listdir(checkpoint_dir):
        match_name = re.match(pattern, filename)
        if match_name:
            iter_num = int(match_name.group(1))
            checkpoints.append((iter_num, filename))

    if not checkpoints:
        return False, None, None

    latest_iter, latest_file = max(checkpoints, key=lambda x: x[0])
    latest_path = os.path.join(checkpoint_dir, latest_file)

    if min_iterations is not None and latest_iter < min_iterations:
        return False, latest_iter, latest_path

    return True, latest_iter, latest_path


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

    min_iterations = solver_args.num_epochs * len(trainloader)

    is_trained, latest_iter, latest_checkpoint = get_latest_checkpoint(
        checkpoint_dir=solver_args.save_path, min_iterations=min_iterations
    )

    if is_trained:
        print(f"Found existing model at iteration {latest_iter}. Loading checkpoint...")
        red_cnn_solver.load_model(latest_iter)
    else:
        if latest_checkpoint:
            print(f"Found partial training checkpoint at iteration {latest_iter}. Resuming training...")
            red_cnn_solver.load_model(latest_iter)
        else:
            print("No existing checkpoints found. Starting training from scratch...")
        red_cnn_solver.train()

    # Change the data_loader to validloader and test to validate
    print("Running validation...")
    red_cnn_solver.data_loader = validloader
    red_cnn_solver.test()


if __name__ == "__main__":
    main()