#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --nodelist=rmc-gpu18
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=clutter6d
#SBATCH --array=1-8
#SBATCH --chdir=/home/grei_lo/Projects/clutter6d
#SBATCH --output=/home/grei_lo/Projects/clutter6d/job_logs/slurm-%j.out
#SBATCH --error=/home/grei_lo/Projects/clutter6d/job_logs/slurm-%j.err

sleep $(( (SLURM_ARRAY_TASK_ID - 1) * 1 ))

# Set output directory based on array task ID
OUTPUT_DIR="/volume/hot_storage/slurm_data/grei_lo/clutter6d_${SLURM_ARRAY_TASK_ID}"

blenderproc run /home/grei_lo/Projects/clutter6d/generate_dataset.py \
  --config config.yml \
  --output_dir "$OUTPUT_DIR"
