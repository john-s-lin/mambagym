import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt
import odl
from scipy.optimize import minimize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def admm_ldct_p3(forward, adjoint, sino, rho, imageResolution, denoise_resolution, model, num_iters=75):

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

def admm_ldct_red(forward, adjoint, sino, rho, lam, imageResolution, denoise_resolution, model, num_iters=50, num_denoise_steps=5):

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

            v = x+u
            input_min, input_max = v.min(), v.max()
            normalized_input = (v - input_min) / (input_max - input_min + 1e-8)
            v_tensor = torch.reshape(torch.from_numpy(normalized_input).float().to(device), (1, 1, v.shape[0], v.shape[1]))
            padding = ((0, denoise_resolution[0] - v.shape[0]), (0, denoise_resolution[0] - v.shape[1]))
            v_padded = torch.nn.functional.pad(v_tensor, (0, padding[1][1], 0, padding[0][1]))

            v_opt = v_padded
            for _ in range(num_denoise_steps):
                v_denoised = model(v_opt)
                v_opt = 1/(rho + lam)*(lam*v_denoised + rho*v_padded)

            v_denoised = v_opt[:, :, :v.shape[0], :v.shape[1]] * (input_max - input_min + 1e-8) + input_min
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


def admm_tv(forward, adjoint, sino, lam, rho, imageResolution, num_iters=75):
    """
    ADMM algorithm for low-dose CT reconstruction with TV prior.

    Parameters:
        forward: Function for applying the forward operator.
        adjoint: Function for applying the adjoint operator.
        sino: Sinogram (measured data).
        rho: ADMM penalty parameter.
        imageResolution: Resolution of the reconstructed image (tuple of ints).
        denoise_resolution: Ignored for TV prior but kept for consistency.
        model: Ignored for TV prior but kept for consistency.
        num_iters: Number of ADMM iterations.

    Returns:
        x: Reconstructed image as a NumPy array.
    """
    # Initialize x, z, and u with zeros
    x = np.zeros(imageResolution)
    z = np.zeros((*imageResolution, 2))  # z stores (z_x, z_y) gradients
    u = np.zeros((*imageResolution, 2))  # u stores dual variables for (z_x, z_y)
    
    img_size = np.prod(imageResolution)

    def gradient(v):
        """Compute forward difference gradients."""
        dx = np.roll(v, -1, axis=1) - v
        dy = np.roll(v, -1, axis=0) - v
        return np.stack((dx, dy), axis=-1)

    def divergence(grad):
        """Compute divergence (transpose of gradient)."""
        dx, dy = grad[..., 0], grad[..., 1]
        dx_roll = np.roll(dx, 1, axis=1)
        dy_roll = np.roll(dy, 1, axis=0)
        return dx_roll - dx + dy_roll - dy

    # Iterative ADMM process
    for it in tqdm(range(num_iters), desc="ADMM Iterations"):
        # Step 1: x-update using Conjugate Gradient (CG) solver
        v = z - u
        cg_iters = 25           # Number of CG iterations
        cg_tolerance = 1e-12    # Convergence tolerance for CG solver

        # Define the A_tilde operator for the linear system
        def a_tilde(u_flat):
            u_reshaped = u_flat.reshape(imageResolution)
            result = adjoint(forward(u_reshaped)) + rho * u_reshaped
            return result.asarray().flatten()
        
        b_tilde = adjoint(sino) + rho * divergence(v)
        b_tilde_flat = b_tilde.asarray().flatten()
        a_tilde_op = LinearOperator((img_size, img_size), matvec=a_tilde)

        # Solve for x using CG
        x_flat, _ = cg(a_tilde_op, b_tilde_flat, rtol=cg_tolerance, maxiter=cg_iters)
        x = x_flat.reshape(imageResolution)

        # Step 2: z-update using anisotropic TV minimization
        kappa = lam/rho
        v = gradient(x) + u
        z = np.maximum(0, 1 - kappa / (np.abs(v) + 1e-8)) * v

        # Step 3: u-update (dual variable update)
        u = u + gradient(x) - z

    return x

def admm_poisson(forward, adjoint, sino, rho, imageResolution, denoise_resolution, model, num_iters=75, gd_max_iters=50):
    """
    Solve Poisson denoising with ADMM using gradient descent for x-update.
    Args:
        forward: Forward operator (function, works on image dimensions).
        adjoint: Adjoint operator (function, works on sinogram dimensions).
        sino: Observed sinogram (original dimensions).
        rho: ADMM penalty parameter.
        imageResolution: Resolution of the output image.
        denoise_resolution: Resolution expected by the denoiser.
        model: Deep learning model for denoising.
        num_iters: Number of ADMM iterations.
        gd_max_iters: Maximum iterations for gradient descent solver.

    Returns:
        Reconstructed image.
    """
    def poisson_objective(x, forward, b, rho, v):
        """
        Poisson objective function for x-update.
        Args:
            x: Current solution (imageResolution).
            forward: Forward operator (function).
            b: Observed sinogram.
            rho: ADMM penalty parameter.
            v: Auxiliary variable (imageResolution).

        Returns:
            Scalar objective value.
        """
        Px = forward(x).asarray().flatten() 
        log_term = -np.sum(b.flatten() * np.log(Px + 1e-8))
        linear_term = np.sum(Px)
        regularization = (rho / 2) * np.linalg.norm(x.flatten() - v.flatten()) ** 2
        return log_term + linear_term + regularization

    def poisson_gradient(x, forward, adjoint, b, rho, v):
        """
        Gradient of the Poisson objective function for x-update.
        Args:
            x_flat: Current solution (flattened).
            forward: Forward operator (function).
            adjoint: Adjoint operator (function).
            b: Observed sinogram.
            rho: ADMM penalty parameter.
            v_flat: Auxiliary variable (flattened).

        Returns:
            Gradient (flattened).
        """
        Px = forward(x) 
        b_over_Px = b / (Px + 1e-8)
        grad = -adjoint(b_over_Px).asarray().flatten() + adjoint(np.ones_like(b)).asarray().flatten() + rho * (x.flatten() - v.flatten())
        return grad

    # Initialize x, z, u to non-zero for numerical stability
    x = np.full(imageResolution, 0.5)
    z = np.full(imageResolution, 0.5)
    u = np.full(imageResolution, 0.5)

    for it in tqdm(range(num_iters), desc="ADMM Iterations"):
        # x-update: Gradient descent
        v = z - u

        def objective(x_flat):
            x_reshaped = x_flat.reshape(imageResolution)
            return poisson_objective(x_reshaped, forward, sino.numpy(), rho, v)

        def gradient(x_flat):
            x_reshaped = x_flat.reshape(imageResolution)
            grad = poisson_gradient(x_reshaped, forward, adjoint, sino.numpy(), rho, v)
            return grad.flatten()

        result = minimize(
            fun=objective,
            x0=x.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': gd_max_iters, 'disp': False}
        )
        x = result.x.reshape(imageResolution)

        # z-update: Denoising step
        v = x + u
        input_min, input_max = v.min(), v.max()
        normalized_input = (v - input_min) / (input_max - input_min + 1e-8)
        v_tensor = torch.reshape(
            torch.from_numpy(normalized_input).float().to(device),
            (1, 1, imageResolution[0], imageResolution[1])
        )

        if denoise_resolution != imageResolution:
            padding = (
                (0, denoise_resolution[0] - imageResolution[0]),
                (0, denoise_resolution[1] - imageResolution[1])
            )
            v_padded = torch.nn.functional.pad(v_tensor, (0, padding[1][1], 0, padding[0][1]))
            v_denoised_padded = model(v_padded)
            v_denoised = v_denoised_padded[:, :, :imageResolution[0], :imageResolution[1]] * (input_max - input_min + 1e-8) + input_min
        else:
            v_denoised = model(v_tensor) * (input_max - input_min + 1e-8) + input_min

        z = torch.squeeze(v_denoised).cpu().numpy()

        # u-update: Dual variable update
        u = u + x - z

    return x