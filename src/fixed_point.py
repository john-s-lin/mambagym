import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse.linalg import cg, LinearOperator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fixed_point_ldct_red(forward, adjoint, sino, sigma_sq, lam, image_resolution, 
                         denoise_resolution, model, num_iters=50, inner_iters=25):
    """
    Fixed-point strategy for RED 
    """
    x = np.zeros(image_resolution)
    model.eval()
    img_size = np.prod(image_resolution)
    
    with torch.no_grad():
        for it in tqdm(range(num_iters), desc="Fixed Point"):
            input_min, input_max = x.min(), x.max()
            normalized_input = (x - input_min) / (input_max - input_min + 1e-8)
            x_tensor = torch.reshape(torch.from_numpy(normalized_input).float().to(device), 
                                  (1, 1, x.shape[0], x.shape[1]))
            
            if denoise_resolution != image_resolution:
                padding = ((0, denoise_resolution[0] - x.shape[0]), 
                          (0, denoise_resolution[1] - x.shape[1]))
                x_padded = torch.nn.functional.pad(x_tensor, 
                                                 (0, padding[1][1], 0, padding[0][1]))
                x_denoised_padded = model(x_padded)
                x_denoised = x_denoised_padded[:, :, :x.shape[0], :x.shape[1]]
            else:
                x_denoised = model(x_tensor)
                
            x_tilde = torch.squeeze(x_denoised).cpu().numpy() * (input_max - input_min + 1e-8) + input_min
            
            def a_operator(v_flat):
                v_reshaped = v_flat.reshape(image_resolution)
                fwd = forward(v_reshaped)
                adj = adjoint(fwd)
                result = (1 / sigma_sq) * adj.asarray() + lam * v_reshaped
                return result.flatten()
            
            b = (1 / sigma_sq) * adjoint(sino).asarray() + lam * x_tilde
            b_flat = b.flatten()
            
            a_op = LinearOperator((img_size, img_size), a_operator)
            
            x_flat, _ = cg(a_op, b_flat, x0=x.flatten(), 
                          maxiter=inner_iters, 
                          rtol=1e-12)
            
            x = x_flat.reshape(image_resolution)
            
    return x