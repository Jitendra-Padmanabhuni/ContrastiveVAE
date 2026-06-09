# Setup & Run Instructions (Yale Bouchet HPC)

End-to-end guide for setting up the environment, downloading the dataset, and running ContrastiveVAE training on Yale's **Bouchet** cluster via Slurm.

---

## Prerequisites

- SSH access to Bouchet (`ssh netid@bouchet.ycrc.yale.edu`)
- A project directory in your home folder (e.g. `~/factor_vae/`) containing `main.py` and the model code
- Familiarity with basic Slurm commands (`sbatch`, `salloc`, `squeue`)

---

## 1. Environment Setup

Run these on the **login node**. The environment only needs to be created once.

```bash
# Load the Conda module
module load miniconda

# Create and activate a dedicated environment
conda create -n factor_env python=3.9 -y
conda activate factor_env

# Install CUDA-enabled PyTorch (CUDA 11.8 build) + core scientific stack
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install pyqlib pandas numpy scipy
```

> **Note:** PyTorch is installed from the official CUDA 11.8 wheel index so it matches the GPU drivers on the Bouchet `gpu` partition. Do **not** `pip install torch` from the default PyPI — it will pull a CPU-only build.

---

## 2. Download the Dataset

Dataset downloads should **not** run on the login node. Request a short interactive session instead.

```bash
# Request an interactive node (1 hour, 8GB RAM, 2 CPUs)
salloc --time=01:00:00 --mem=8G --cpus-per-task=2

# Wait for the shell prompt to switch to a compute node, then:
module load miniconda
conda activate factor_env

# Download the Qlib US equity dataset (2000–2020) into your home directory
python -c "from qlib.tests.data import GetData; GetData().qlib_data(target_dir='~/.qlib/qlib_data/us_data', region='us')"

# Release the interactive node when the download finishes
exit
```

The dataset lands in `~/.qlib/qlib_data/us_data/` and only needs to be downloaded once.

---

## 3. Slurm Job Script

From your project directory, create the submission script:

```bash
nano run_factor.sh
```

Paste the following:

```bash
#!/bin/bash
#SBATCH --job-name=factor_vae
#SBATCH --output=factor_vae_%j.out   # Stdout + stderr log (%j = job ID)
#SBATCH --partition=gpu              # GPU partition
#SBATCH --gpus=1                     # 1 GPU
#SBATCH --cpus-per-task=8            # 8 CPU cores for data loading
#SBATCH --mem=64G                    # 64 GB RAM for the 158-feature batches
#SBATCH --time=12:00:00              # Safely under partition limits

# 1. Load modules and activate the environment
module load miniconda
conda activate factor_env

# 2. (Optional) cd into your project directory
# cd ~/factor_vae

# 3. Run training
echo "Starting ContrastiveVAE training on Bouchet GPU..."
python main.py
```

Save and exit nano: `Ctrl + O`, `Enter`, then `Ctrl + X`.

### Evaluation-only mode

To skip training and just compute Rank IC / run the backtest on a saved checkpoint, change the last line of the script to:

```bash
python main.py --mode eval
```

---

## 4. Submit & Monitor

```bash
# Submit the job
sbatch run_factor.sh

# Check queue status for your user
squeue -u $USER

# Tail the live log
tail -f factor_vae_*.out

# Cancel a running job if needed
scancel <job_id>
```

When the job finishes, the full log persists at `factor_vae_<JOBID>.out` in the working directory.

---

## Resource Notes

| Setting | Value | Why |
|---|---|---|
| `--partition=gpu` | gpu | ContrastiveVAE training is GPU-bound (Transformer + InfoNCE) |
| `--gpus=1` | 1 | Model fits comfortably on a single GPU; multi-GPU not yet wired up |
| `--cpus-per-task=8` | 8 | Parallelizes the Qlib data loader and view-sampling step |
| `--mem=64G` | 64 GB | Alpha158 features × full US cross-section × batch of trading days |
| `--time=12:00:00` | 12 h | Comfortably above measured ~6–8 h full-training time |

If your job is queued for a long time, consider lowering `--time` or `--mem` to fit smaller scheduling windows.

---

## Troubleshooting

- **`CUDA out of memory`** — reduce batch size in `main.py`, or lower the cross-section subsample size used by the CSCL view generator.
- **Conda environment not found on compute node** — make sure you `module load miniconda` *inside* the Slurm script; logged-in modules don't carry over.
- **Qlib download hangs / fails** — re-run the interactive session step; Yahoo Finance occasionally rate-limits bulk requests.
- **Job pending forever** — check `squeue -u $USER` for the reason code; `Resources` means you're waiting on a GPU, `QOSMaxJobsPerUser` means you've hit a concurrency cap.
