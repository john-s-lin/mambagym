import dival
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as rmse

import numpy as np
import odl
from tqdm import tqdm
from torch.utils.data import Subset
from datetime import datetime

if __name__ == "__main__":
    # Load dataset
    dataset = dival.datasets.get_standard_dataset('lodopab')
    data = dataset.create_torch_dataset(part='train')
    data = Subset(data, indices=range(200))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Obtain the ray transform and the FBP reconstruction operator
    ray_transform = dataset.ray_trafo
    fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform)
    psnrs = []
    ssims = []
    rmses = []

    for batch_number, (sino, gt) in tqdm(enumerate(data)):
        # Perform FBP reconstruction
        reconstruction = fbp_operator(sino.numpy()).asarray()

        # Compute metrics
        _psnr = psnr(gt.numpy(), reconstruction)
        _ssim = ssim(gt.numpy(), reconstruction, data_range=np.max(gt.numpy()) - np.min(gt.numpy()))
        _rmse = rmse(gt.numpy(), reconstruction)
        psnrs.append(_psnr)
        ssims.append(_ssim)
        rmses.append(_rmse)

        # Save visualizations every 10 batches
        if batch_number % 10 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(reconstruction, cmap='gray')
            axes[0].set_title('Reconstruction')
            axes[0].axis('off')
            psnr_ssim_text = f"PSNR: {_psnr:.2f} dB\nSSIM: {_ssim:.4f}\nRMSE: {_rmse:.4f}"
            axes[0].text(
                0.05, 0.05, psnr_ssim_text,
                transform=axes[0].transAxes,
                fontsize=10, color='white',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')
            )
            axes[1].imshow(gt.numpy(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            output_path = f"plots/fbp/img_{batch_number}_comparison_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.imsave(f"plots/fbp/img_{batch_number}_reconstruction_{timestamp}.png", reconstruction, cmap='gray')

    # Compute and save overall metrics
    print(np.mean(psnrs))
    print(np.mean(ssims))
    print(np.mean(rmses))
    np.save(f"plots/fbp/psnrs_{timestamp}.npy", np.array(psnrs))
    np.save(f"plots/fbp/ssims_{timestamp}.npy", np.array(ssims))
    np.save(f"plots/fbp/rmses_{timestamp}.npy", np.array(rmses))
