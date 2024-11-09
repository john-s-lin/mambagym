# High-Performance Computing on `slurm` Clusters

## Premise

Using high-performance clusters with `slurm` gives us access to a myriad of high-performance CPUs and GPUs on which to run intensive machine learning programs in order to reduce training times.

## Steps

### 1. Ensure your virtual environment is activated and all dependencies are installed.

If you are using `venv`, or `uv venv`, activate your virtual environment first.

```bash
source <path-to-virtual-environment>/bin/activate
```

If you are using `conda`, activate your `conda` environment first.

```bash
# E.g. environment-name = 'mambagym`
conda activate <environment-name>
```

#### Dependencies

With `uv`, you can use `uv sync` to draw all dependencies from the `uv.lock` lockfile.

```bash
uv sync
```

With `venv`, you can use `pip` on a `requirements.txt` or `pyproject.toml` file.

```bash
pip install . # Preferred if pyproject.toml exists
pip install -r <path-to-requirements.txt> # If requirements.txt exists
```

With `conda`, you can use `pip` for `requirements.txt` and `conda` for `environment.yml`. The preferred way to install dependencies for `conda` is with `environment.yml`.

```bash
conda env create -f <path-to-environment.yml>

# Update an existing env with packages from environment.yml
conda env update -f <path-to-environment.yml> -n <environment-name>

# If there are dependencies on pip but not in conda
pip install -r <path-to-requirements.txt>
```

### 2. Prepare a `bash` script as an entrypoint to run your program

> Why not use `srun`? Because we can ensure repeatibility by declaring our requirements in the bash script beforehand, and save ourselves the trouble of typing out our required resources each time.

Let's say you have a `bash` script called `entrypoint.sh`. We have to explicitly declare our required resources in comments in this `bash` script, otherwise the control node will randomly assign our job to a general-purpose CPU node with the minimum number of resources.

In `entrypoint.sh`:

```bash
#!/usr/bin/env bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1 # This gives you the first available gpu, but you can specify which one you want
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

python src/<path-to-your-script.py>
```

> Make sure your project is in the filepath with lots of available space, such as the `/w/340/<your-name>` directory.

### 3. Submit your job with `sbatch`

The benefit of using `sbatch` is you can run your script heterogeneously, meaning you can run your job with a `gpunode` cluster with `16G` of memory and 4 cores as well as a `cpunode` cluster with `64G` of memory and 16 cores if needed. For further information refer to [these docs](https://slurm.schedmd.com/heterogeneous_jobs.html).

Then submit your job with `sbatch` from a `comps[0-3]` server:

```bash
sbatch entrypoint.sh
```

## Future work

Connecting to a `gpunode` to run a `jupyter notebook` server is much more intensive but doable. However, there is significant compute overhead with running a notebook, which is essentially a parsed `json` file.
