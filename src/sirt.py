import numpy as np
def sirt(image_shape, sinogram, forward, adjoint, num_iterations=100, lambda_param=0.1):
    # Normalization factor in sinogram space
    uniform_image = np.ones(image_shape, dtype=np.float32)
    sinogram_norm = forward(uniform_image)
    sinogram_norm[sinogram_norm == 0] = 1
    # Normalization factor in image space
    uniform_sinogram = np.ones_like(sinogram)
    image_norm = adjoint(uniform_sinogram)
    image_norm[image_norm == 0] = 1 

    reconstruction = np.zeros(image_shape)
    for _ in range(num_iterations):
        proj_estimate = forward(reconstruction)
        error_sinogram = (sinogram - proj_estimate)/sinogram_norm
        backprojected_error = adjoint(error_sinogram)/image_norm
        # Update the reconstruction (scaled back-projected error)
        reconstruction += lambda_param * backprojected_error
        
    return reconstruction