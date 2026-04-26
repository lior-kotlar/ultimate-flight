#!/bin/bash
#SBATCH --job-name=inverse_mapping_train
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

# Usage: sbatch -J <EXPERIMENT_NAME> sbatch_train_inverse_mapping.sh <CONFIG_PATH>
CONFIG_PATH=$1

# Safety checks
if [ -z "$CONFIG_PATH" ]; then
  echo "Error: No config file path provided."
  echo "Usage: sbatch -J <EXPERIMENT_NAME> sbatch_train_inverse_mapping.sh path/to/config.json"
  exit 1
fi

echo "started"

# Navigate to the correct ultimate-flight workspace
cd /cs/labs/tsevi/lior.kotlar/ultimate-flight

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the virtual environment
source .env/bin/activate

echo "Job started on $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Config File: $CONFIG_PATH"
echo "Experiment Name: $SLURM_JOB_NAME"

# Execute the python script
python code/train_inverse_mapping.py --config "$CONFIG_PATH" --name "$SLURM_JOB_NAME"

echo "finished working"