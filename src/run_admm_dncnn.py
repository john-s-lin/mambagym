import dival
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as rmse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from admm_denomamba import admm_ldct 
from datetime import datetime
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from models.dncnn.network_dncnn import DnCNN as net

if __name__ == "__main__":
    dataset = dival.datasets.get_standard_dataset('lodopab', impl='astra_cpu')
    data = dataset.create_torch_dataset(part='train')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = Subset(data, indices=range(1))
    originial_transform = dataset.ray_trafo
    reco_space = originial_transform.domain
    geometry = originial_transform.geometry
    angles = geometry.angles
    psnrs = []
    ssims = []
    rmses = []
    denoiser = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
    denoiser.load_state_dict(torch.load('/u/yukthiw/files/mambagym/src/models/dncnn/dncnn_25.pth'), strict=True)
    denoiser.eval()

    with torch.no_grad():
        for batch_number, (sino, gt) in tqdm(enumerate(data)):
            reconstruction = admm_ldct(originial_transform, originial_transform.adjoint, sino, 1.0, (362, 362), denoise_resolution=(362, 362), model=denoiser, num_iters=75)
            _psnr = psnr(gt.numpy(), reconstruction)
            _ssim = ssim(gt.numpy(), reconstruction, data_range=np.max(gt.numpy()) - np.min(gt.numpy()))
            _rmse = rmse(gt.numpy(), reconstruction)
            psnrs.append(_psnr)
            ssims.append(_ssim)
            rmses.append(_rmse)
            if batch_number % 10 == 0:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(reconstruction, cmap='gray')
                axes[0].set_title('Reconstruction')
                axes[0].axis('off')
                psnr_ssim_text = f"PSNR: {_psnr:.2f} dB\nSSIM: {_ssim:.4f}\RMSE:: {_rmse:.4f}"
                axes[0].text(
                    0.05, 0.05, psnr_ssim_text,
                    transform=axes[0].transAxes,
                    fontsize=10, color='white',
                    bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')
                )
                axes[1].imshow(gt.numpy(), cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                output_path = f"plots/img_{batch_number}_comparison_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plt.imsave(f"plots/admm_dncnn/img_{batch_number}_reconstruction_{timestamp}.png", reconstruction)
    print(np.mean(psnrs))
    print(np.mean(ssims))
    print(np.mean(rmses))
    np.save(f"plots/admm_dncnn/psnrs_{timestamp}.npy",np.array(psnrs))
    np.save(f"plots/admm_dncnn/ssims_{timestamp}.npy",np.array(ssims))
    np.save(f"plots/admm_dncnn/rmses_{timestamp}.npy",np.array(rmses))
