#!/bin/bash
# ============================================================
# SLURM job — Task 1: Google Stock Price Prediction
#   Trains RNN, LSTM, and GRU models sequentially.
#
# Submit from the assignment3/ directory:
#   sbatch jobs/task1.sh
#
# Prerequisites:
#   mkdir -p logs checkpoints results
# ============================================================

#SBATCH --job-name=nn_a3_task1
#SBATCH --output=logs/task1_%j.out
#SBATCH --error=logs/task1_%j.err
#SBATCH --partition=compute          # <-- update to your cluster's CPU partition
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=228G

# ---- user-configurable variables ------------------------------------
DATA_DIR=~/neural-networks/assignment3/data
EPOCHS=100
BATCH_SIZE=32
LR=1e-3
TIMESTEPS=60
SEED=42
# ---------------------------------------------------------------------

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Task       : 1 — Stock Price Prediction"
echo "Start time : $(date)"
echo "========================================"

module load Python
module load SciPy-bundle

# Copy dataset to local scratch for faster I/O
SCRATCH_DATA_DIR="${TMPDIR}/data"
mkdir -p "${SCRATCH_DATA_DIR}"
cp "${DATA_DIR}"/Google_Stock_Price_*.csv "${SCRATCH_DATA_DIR}/"
echo "Data copied to scratch: $(date)"

cd "${SLURM_SUBMIT_DIR}" || exit 1

python task1.py \
    --data_dir       "${SCRATCH_DATA_DIR}" \
    --results_dir    results \
    --checkpoints_dir checkpoints \
    --epochs         "${EPOCHS}" \
    --batch_size     "${BATCH_SIZE}" \
    --lr             "${LR}" \
    --timesteps      "${TIMESTEPS}" \
    --seed           "${SEED}" \
    --num_threads    48

echo "Done: Task 1  end=$(date)"
