#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --nodelist=rmc-gpu18
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=clutter6d
#SBATCH --array=1-5
#SBATCH --chdir=/home/grei_lo/Projects/clutter6d
#SBATCH --output=/home/grei_lo/Projects/clutter6d/job_logs/slurm-%j.out
#SBATCH --error=/home/grei_lo/Projects/clutter6d/job_logs/slurm-%j.err

sleep $(( (SLURM_ARRAY_TASK_ID - 1) * 1 ))

# Set output directory based on array task ID
OUTPUT_DIR="/volume/hot_storage/slurm_data/grei_lo/extra_run/clutter6d_${SLURM_ARRAY_TASK_ID}"

# Run the rendering script multiple times to prevent memory issues
for i in {1..10}; do
    echo "Starting run $i/10 for job ${SLURM_ARRAY_TASK_ID}"
    
    blenderproc run /home/grei_lo/Projects/clutter6d/generate_dataset.py \
      --config config_slurm_25.yml \
      --output_dir "$OUTPUT_DIR"
    
    echo "Completed run $i for job ${SLURM_ARRAY_TASK_ID}"
    sleep 5
done

echo "All runs completed for job ${SLURM_ARRAY_TASK_ID}"
