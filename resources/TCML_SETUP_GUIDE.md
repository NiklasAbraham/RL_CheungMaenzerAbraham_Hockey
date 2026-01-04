# TCML Cluster Setup Guide for Hyperparameter Training

This guide explains how to run the hyperparameter training script on the TCML cluster.

## Step 1: Upload Your Project

From your local machine, upload the project to the cluster:

```bash
# Upload the entire project directory
scp -r /path/to/RL_CheungMaenzerAbraham_Hockey USERNAME@login1.tcml.uni-tuebingen.de:~/
```

Or use `rsync` for better efficiency:

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    /path/to/RL_CheungMaenzerAbraham_Hockey \
    USERNAME@login1.tcml.uni-tuebingen.de:~/
```

## Step 2: Log in to the Cluster

```bash
ssh USERNAME@login1.tcml.uni-tuebingen.de
```

If login1 doesn't work, try login2 or login3.

## Step 3: Build the Singularity Container

On the login node, navigate to your project directory and build the container:

```bash
cd ~/RL_CheungMaenzerAbraham_Hockey
mkdir -p singularity_build
cd singularity_build

# Copy the container definition
cp ../resources/container/container.def ./rl_hockey.def

# Build the container image
singularity build --fakeroot rl_hockey.simg rl_hockey.def
```

**Note:** The container build may take 10-30 minutes depending on network speed for downloading packages.

## Step 4: Install the rl_hockey Package in the Container (Optional but Recommended)

You can create a sandbox version of the container to install your package:

```bash
# Create a sandbox from the image
singularity build --fakeroot --sandbox rl_hockey_sandbox rl_hockey.simg

# Open a shell in the sandbox
singularity shell --writable --bind ~/RL_CheungMaenzerAbraham_Hockey:~/RL_CheungMaenzerAbraham_Hockey rl_hockey_sandbox

# Inside the container, install the package
cd ~/RL_CheungMaenzerAbraham_Hockey
pip install -e .

# Exit the container
exit

# Build the final image from the sandbox
singularity build --fakeroot rl_hockey_final.simg rl_hockey_sandbox
```

Alternatively, you can install the package at runtime in the .sbatch file (see Step 5).

## Step 5: Configure the .sbatch File

Edit the `hyperparameter_training.sbatch` file:

1. **Update the email address:**
   ```bash
   #SBATCH --mail-user=YOUR_EMAIL@uni-tuebingen.de
   ```

2. **Update the container path:**
   ```bash
   SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey.simg"
   ```
   Or if you built the final version with the package installed:
   ```bash
   SINGULARITY_IMAGE="$HOME/RL_CheungMaenzerAbraham_Hockey/singularity_build/rl_hockey_final.simg"
   ```

3. **Update the project directory path:**
   ```bash
   PROJECT_DIR="$HOME/RL_CheungMaenzerAbraham_Hockey"
   ```

4. **If the package is not installed in the container, add installation step:**
   Add this line before running the script:
   ```bash
   # Install the package in editable mode
   singularity exec --nv --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$PROJECT_DIR" \
       "$SINGULARITY_IMAGE" \
       pip install -e "$PROJECT_DIR"
   ```

## Step 6: Adjust Resource Requirements (if needed)

The default .sbatch file requests:
- 8 CPUs
- 32GB RAM (4GB per CPU)
- 1 GPU
- 7 days runtime (week partition)

You can adjust these based on your needs:
- For faster training with more parallel workers, you can request more GPUs (up to 4 per node for 1080ti/A4000, or 8 for 2080ti/L40S)
- For more memory-intensive configurations, increase `--mem-per-cpu`
- For shorter jobs, use `--partition=day` with `--time=24:00:00`

## Step 7: Submit the Job

```bash
cd ~/RL_CheungMaenzerAbraham_Hockey
sbatch hyperparameter_training.sbatch
```

You'll receive a job ID. Note it down for tracking.

## Step 8: Monitor Your Job

```bash
# Check job status
squeue -u $USER

# Check detailed job information
squeue --start -j JOBID

# Cancel a job if needed
scancel JOBID
```

## Step 9: Check Results

Once the job completes, check the output files:

```bash
# View output
cat job.JOBID.out

# View errors (if any)
cat job.JOBID.err

# Check results directory
ls -lh results/hyperparameter_runs/
```

## Important Notes

1. **Storage:** Keep your home directory small. Move large result files to scratch space if available, or download them to your local machine.

2. **Time Limits:** The week partition has a 7-day limit. If your job needs longer, use the month partition (30 days).

3. **GPU Selection:** By default, the job will use any available GPU. To request a specific GPU type, modify the `--gres` line:
   ```bash
   #SBATCH --gres=gpu:1080ti:1
   #SBATCH --gres=gpu:2080ti:1
   #SBATCH --gres=gpu:A4000:1
   #SBATCH --gres=gpu:L40S:1  # Only if you have access
   ```

4. **Parallel Workers:** The script uses 4 parallel workers by default. You can adjust this in the .sbatch file or pass `--num_parallel N` to the script.

5. **Data Location:** Make sure any datasets or data files are accessible from the compute nodes. If using shared storage, mount paths should be consistent.

## Troubleshooting

- **Job not starting:** Check `squeue` to see if resources are available. Your job may be queued.
- **Out of memory:** Increase `--mem-per-cpu` or reduce `--num_parallel` workers.
- **Import errors:** Make sure the rl_hockey package is installed in the container or install it at runtime.
- **CUDA errors:** Ensure `--nv` flag is used in singularity run command to enable GPU access.
