import numpy as np
import odl
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt

def admm_ldct(forward, adjoint, sino, rho, imageResolution, denoise_resolution, model, num_iters=75):

    # initialize x,z,u with all zeros
    x = np.zeros(imageResolution)
    z = np.zeros(imageResolution)
    u = np.zeros(imageResolution)
    
    model.eval()
    img_size = np.prod(imageResolution)

    with torch.no_grad():
        for it in tqdm(range(num_iters)):
            # x update using cg solver
            v = z-u
            cg_iters = 25           # number of iterations for CG solver
            cg_tolerance = 1e-12    # convergence tolerance of cg solver

            def a_tilde(u_flat):
                # Reshape the flat input vector to 2D image
                u_reshaped = u_flat.reshape(imageResolution)
                # Apply the Afun and Atfun operators, then flatten the result
                result = adjoint(forward(u_reshaped)) + rho * u_reshaped
                return result.asarray().flatten()
            
            b_tilde = adjoint(sino) + rho*v
            b_tilde_flat = b_tilde.asarray().flatten()
            a_tilde_op = LinearOperator((img_size, img_size), a_tilde)

            x_flat, _ = cg(a_tilde_op, b_tilde_flat, rtol=cg_tolerance, maxiter=cg_iters)
            x = x_flat.reshape(imageResolution)
            ################# end task 3 ###################################

            # z-update using DnCNN denoiser
            v = x+u
            input_min, input_max = v.min(), v.max()
            normalized_input = (v - input_min) / (input_max - input_min + 1e-8)
            v_tensor = torch.reshape(torch.from_numpy(normalized_input).float().to(device), (1, 1, v.shape[0], v.shape[1]))
            if denoise_resolution != imageResolution:
                padding = ((0, denoise_resolution[0] - v.shape[0]), (0, denoise_resolution[0] - v.shape[1]))
                v_padded = torch.nn.functional.pad(v_tensor, (0, padding[1][1], 0, padding[0][1]))
                v_denoised_padded = model(v_padded)
                v_denoised = v_denoised_padded[:, :, :v.shape[0], :v.shape[1]] * (input_max - input_min + 1e-8) + input_min
            else:
                v_denoised = model(v_tensor) * (input_max - input_min + 1e-8) + input_min
            z = torch.squeeze(v_denoised).cpu().numpy()

            # u update
            u = u+x-z

    return x


def admm_ldct_notorch(forward, adjoint, sino, rho, imageResolution, model, num_iters=75):

    # initialize x,z,u with all zeros
    x = np.zeros(imageResolution)
    z = np.zeros(imageResolution)
    u = np.zeros(imageResolution)
    
    img_size = np.prod(imageResolution)

    for it in tqdm(range(num_iters)):
        # x update using cg solver
        v = z-u
        cg_iters = 25           # number of iterations for CG solver
        cg_tolerance = 1e-12    # convergence tolerance of cg solver

        def a_tilde(u_flat):
            # Reshape the flat input vector to 2D image
            u_reshaped = u_flat.reshape(imageResolution)
            # Apply the Afun and Atfun operators, then flatten the result
            result = adjoint(forward(u_reshaped)) + rho * u_reshaped
            return result.asarray().flatten()
        
        b_tilde = adjoint(sino) + rho*v
        b_tilde_flat = b_tilde.asarray().flatten()
        a_tilde_op = LinearOperator((img_size, img_size), a_tilde)

        x_flat, _ = cg(a_tilde_op, b_tilde_flat, rtol=cg_tolerance, maxiter=cg_iters)
        x = x_flat.reshape(imageResolution)
        ################# end task 3 ###################################

        # z-update using DnCNN denoiser
        v = x+u
        v_denoised = model(v)
        z = v_denoised
        plt.imsave("test4.png", v)
        plt.imsave("test5.png", v_denoised)
        # u update
        u = u+x-z

    return x