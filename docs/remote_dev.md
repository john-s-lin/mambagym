# Remote development with SSH on Visual Studio Code

This is the recommended way to develop for this project, for the following reasons:

- Access to high-performance compute including access to `cuda`-enabled Nvidia GPUs.
- Access to `slurm`

However, it is not without drawbacks, which include:

- A requirement to install a `conda` environment locally, which requires additional work and **LOTS** of space, given the `nvidia-cuda` packages.
- Inability to maintain consistent developer environments

## Setup

### 0. Destinations

In the `comps[0-3].cs.toronto.edu` servers, you may not have enough space to install `conda` _and_ the required packages for this project in your home directory. You can install `conda` in your home directory, but you will have to set a special prefix to install the `mambagym` environment in a `/w/` directory with more space.

These directories should be available:

- `/w/246`
- `/w/247`
- `/w/284`
- `/w/331`
- `/w/339`
- `/w/340`
- `/w/383`

For this example, let's say we'll use the directory `/w/383/<your-cs-toronto-username>`.

### 1. Install `miniforge` for `conda`

Here we'll use the [`miniforge`](https://github.com/conda-forge/miniforge) installation. `miniforge` is a minimal instance of `Anaconda`, with the default channel being `conda-forge`. Additionally, `mamba`, the faster dependency resolver for `conda`, touted as a drop-in replacement, has been deprecated since the resolver has been integrated into `miniconda` and `miniforge` directly since [Release 23.10.0](https://docs.conda.io/projects/conda/en/latest/release-notes.html#with-this-23-10-0-release-we-are-changing-the-default-solver-of-conda-to-conda-libmamba-solver) [[Ref.](https://conda-forge.org/news/2024/07/29/sunsetting-mambaforge/)].

If `conda` is not installed yet, you can install the `miniforge3` version in your home directory.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

It shouldn't matter the location of your `miniforge3` installation since in the setup script `conda_setup.sh`, we will be creating a `pkgs_dirs` and `env_dirs` in our target directory, hopefully with much more available space.

### 2. Clone `mambagym` into your target directory

`cd` into your target directory, in our example `/w/383/<your-cs-toronto-username>`, and clone this repo.

```bash
git clone git@github.com:john-s-lin/mambagym.git
```

### 3. Run `conda_setup.sh` in `slurm` to create a new `conda` environment with `cuda` enabled

In the root of your `mambagym` repo, edit `conda_setup.sh` to set your `TARGET_DIR_PREFIX` to your current repo.

```bash
# conda_setup.sh

# /w/383 can be any one of the directories above in 0. Destinations
TARGET_DIR_PREFIX="/w/383/${USER}"
```
