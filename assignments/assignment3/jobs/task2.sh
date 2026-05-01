#!/bin/bash
# ============================================================
# SLURM job — Task 2: Hotel Description Generation
#   Trains LSTM, GRU, and Transformer models sequentially.
#
# Submit from the assignment3/ directory:
#   sbatch jobs/task2.sh
#
# Prerequisites:
#   mkdir -p logs checkpoints results
# ============================================================

#SBATCH --job-name=nn_a3_task2
#SBATCH --output=logs/task2_%j.out
#SBATCH --error=logs/task2_%j.err
#SBATCH --partition=compute          # <-- update to your cluster's CPU partition
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=228G

# ---- user-configurable variables ------------------------------------
DATA_DIR=~/neural-networks/assignment3/data
EPOCHS=50
BATCH_SIZE=128
LR=1e-3
NEXT_WORDS=20
SEED=42
# ---------------------------------------------------------------------

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Task       : 2 — Hotel Description Generation"
echo "Start time : $(date)"
echo "========================================"

module load Python
module load SciPy-bundle

pip install keras tensorflow-cpu --user 2>&1 | tail -1
pip install tensorflow --user 2>&1 | tail -1

# Copy dataset to local scratch for faster I/O
SCRATCH_DATA_DIR="${TMPDIR}/data"
mkdir -p "${SCRATCH_DATA_DIR}"
cp "${DATA_DIR}"/Seattle_Hotels_address_description.csv "${SCRATCH_DATA_DIR}/"
echo "Data copied to scratch: $(date)"

cd "${SLURM_SUBMIT_DIR}" || exit 1

python task2.py \
    --data_dir       "${SCRATCH_DATA_DIR}" \
    --results_dir    results \
    --checkpoints_dir checkpoints \
    --epochs         "${EPOCHS}" \
    --batch_size     "${BATCH_SIZE}" \
    --lr             "${LR}" \
    --next_words     "${NEXT_WORDS}" \
    --seed           "${SEED}" \
    --num_threads    48

echo "Done: Task 2  end=$(date)"
