import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import dival
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import normalized_root_mse as rmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Subset
from tqdm import tqdm

from fixed_point import fixed_point_ldct_red
from models.red_cnn.solver import Solver

"""
NOTE:
For the `dival` package, you will have a `~/.dival/config.json` file created in your
home directory.

In it, you will find a configuration locations for `lodopab` and `reference_params`.
Since we're using `lodopab`, make sure you set this directory to where the
`lodopab` dataset is stored.

For example:
```json
{
 "lodopab_dataset": {
  "data_path": "<target_lodopab_dir>"
 },
 "reference_params": {
  "data_path": "${HOME}/.dival/reference_params"
 }
}
```
"""

MODEL_CHECKPOINT_PATH = Path(__file__).parent.joinpath("models/red_cnn/checkpoints").resolve()
PLOT_SAVE_PATH = Path(__file__).parent.joinpath("plots/fixed_point_red_cnn").resolve()


def find_latest_checkpoint(checkpoint_dir: str, pattern: str) -> str:
    checkpoints = []

    regex = re.compile(pattern)

    for filename in os.listdir(checkpoint_dir):
        match_name = regex.match(filename)
        if match_name is not None:
            iter_num = int(match_name.group(1))
            checkpoints.append((iter_num, filename))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files matching pattern {pattern} found in {checkpoint_dir}")

    latest_iter, latest_file = max(checkpoints, key=lambda x: x[0])
    latest_path = os.path.join(checkpoint_dir, latest_file)

    return latest_path, latest_iter


def init_red_cnn(checkpoint_dir: str) -> Solver:
    # This is a hack to pass in arguments manually to the RED_CNN trainer
    # From models/red_cnn/main.py, using the defaults
    # See src/train_red_cnn.py
    solver_args = argparse.Namespace(
        mode="test",
        load_mode=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        norm_range_min=-1024.0,
        norm_range_max=3072.0,
        trunc_min=-160.0,
        trunc_max=240.0,
        save_path=checkpoint_dir,
        save_fig_path=PLOT_SAVE_PATH,
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

    # You're testing the model on the lodopab dataset
    # so don't need to use the data_loader as you did for training and validation
    model = Solver(solver_args, data_loader=None)

    _, latest_iter = find_latest_checkpoint(checkpoint_dir=checkpoint_dir, pattern=r"REDCNN_(\d+)_epoch.ckpt")
    model.load_model(latest_iter)

    return model


# TODO: set dival config location to /w/383/<USERNAME>/.dival/
# Ref: https://jleuschn.github.io/docs.dival/dival.config.html
def main():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = dival.datasets.get_standard_dataset("lodopab")
    data = dataset.create_torch_dataset("train")
    data = Subset(data, indices=range(1000))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_transform = dataset.ray_trafo

    psnr_list = []
    ssim_list = []
    rmse_list = []

    red_cnn_model = init_red_cnn(MODEL_CHECKPOINT_PATH)

    with torch.no_grad():
        for batch_number, (sino, gt) in tqdm(enumerate(data)):
            reconstruction = fixed_point_ldct_red(
                original_transform,
                original_transform.adjoint,
                sino,
                sigma=1,
                lam=0.1,
                image_resolution=(362, 362),
                denoise_resolution=(512, 512),
                model=red_cnn_model.REDCNN,
                num_iters=50,
                inner_iters=25,
            )
            _psnr = psnr(gt.numpy(), reconstruction)
            _ssim = ssim(gt.numpy(), reconstruction, data_range=np.max(gt.numpy()) - np.min(gt.numpy()))
            _rmse = rmse(gt.numpy(), reconstruction)
            psnr_list.append(_psnr)
            ssim_list.append(_ssim)
            rmse_list.append(_rmse)
            if batch_number % 10 == 0:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(reconstruction, cmap="gray")
                axes[0].set_title("Reconstruction")
                axes[0].axis("off")
                psnr_ssim_text = f"PSNR: {_psnr:.2f} dB\nSSIM: {_ssim:.4f}\nRMSE: {_rmse:.4f}"
                axes[0].text(
                    0.05,
                    0.05,
                    psnr_ssim_text,
                    transform=axes[0].transAxes,
                    fontsize=10,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.3"),
                )
                axes[1].imshow(gt.numpy(), cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                output_path = f"{PLOT_SAVE_PATH}/img_{batch_number}_comparison_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                plt.imsave(f"{PLOT_SAVE_PATH}/img_{batch_number}_reconstruction_{timestamp}.png", reconstruction)
    print(f"Avg. PSNR: {np.mean(psnr_list)}")
    print(f"Avg. SSIM: {np.mean(ssim_list)}")
    print(f"Avg. RMSE: {np.mean(rmse_list)}")
    np.save(f"{PLOT_SAVE_PATH}/psnrs_{timestamp}.npy", np.array(psnr_list))
    np.save(f"{PLOT_SAVE_PATH}/ssims_{timestamp}.npy", np.array(ssim_list))
    np.save(f"{PLOT_SAVE_PATH}/rmses_{timestamp}.npy", np.array(rmse_list))


if __name__ == "__main__":
    if not os.path.exists(PLOT_SAVE_PATH):
        os.makedirs(PLOT_SAVE_PATH)
        print(f"Creating plot directory at {PLOT_SAVE_PATH}")
    else:
        print(f"{PLOT_SAVE_PATH} already exists")
    main()
