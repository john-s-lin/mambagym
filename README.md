# mambagym

Optimizing priors for denoising low-dose CT

## Remote development with SSH (recommended)

For info on how to develop remotely on SSH with `vscode`, refer to [`/docs/remote_dev.md`](./docs/remote_dev.md).

## Environment setup

```
cd mambagym
conda env create .
conda activate mambagym
```

## Running on high-performance clusters with `slurm`

For info on how to run Python training or inference scripts on slurm, refer to [`/docs/slurm.md`](./docs/slurm.md).

### GPU Environment

The available GPUs on the Slurm cluster run CUDA 12.9. Ensure your JAX/PyTorch installations are compatible (e.g., via `pip install jax[cuda12_pip]` for JAX).
