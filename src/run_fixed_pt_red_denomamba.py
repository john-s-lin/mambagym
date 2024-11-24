import dival
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as rmse

import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
import torch
from datetime import datetime
from fixed_point import fixed_point_ldct_red
from models.DenoMamba.denomamba_arch import DenoMamba

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_deno_mamba(weights_path, in_ch=1, out_ch=1, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=2):
    model = DenoMamba(
        inp_channels=in_ch,
        out_channels=out_ch,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
    ).to(device)
    ckpt_model = torch.load(weights_path, map_location=device)
    net_weights = ckpt_model["net_model"]
    model.load_state_dict(net_weights)
    return model

if __name__ == "__main__":
    dataset = dival.datasets.get_standard_dataset('lodopab')
    data = dataset.create_torch_dataset(part='train')
    data = Subset(data, indices=range(10))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_transform = dataset.ray_trafo
    reco_space = original_transform.domain
    geometry = original_transform.geometry
    angles = geometry.angles
    
    psnrs = []
    ssims = []
    rmses = []
    
    denoiser = init_deno_mamba(weights_path="/w/331/yukthiw/checkpoints/model_epoch_60.pkl")

    with torch.no_grad():
        for batch_number, (sino, gt) in tqdm(enumerate(data)):
            reconstruction = fixed_point_ldct_red(
                original_transform, 
                original_transform.adjoint,
                sino,
                sigma=0.1,
                lam=0.1,
                image_resolution=(362, 362),
                denoise_resolution=(512, 512),
                model=denoiser,
                num_iters=50,
                inner_iters=25
            )
            
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
                
                output_path = f"plots/fixed_point_denomamba_red/img_{batch_number}_comparison_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                plt.imsave(f"plots/fixed_point_denomamba_red/img_{batch_number}_reconstruction_{timestamp}.png", 
                          reconstruction, cmap='gray')
                
    print(f"Average PSNR: {np.mean(psnrs):.2f}")
    print(f"Average SSIM: {np.mean(ssims):.4f}")
    print(f"Average RMSE: {np.mean(rmses):.4f}")
    
    np.save(f"plots/fixed_point_denomamba_red/psnrs_{timestamp}.npy", np.array(psnrs))
    np.save(f"plots/fixed_point_denomamba_red/ssims_{timestamp}.npy", np.array(ssims))
    np.save(f"plots/fixed_point_denomamba_red/rmses_{timestamp}.npy", np.array(rmses))