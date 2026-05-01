#!/bin/bash
# ============================================================
# SLURM job — Baseline Unconditional DCGAN (terrain, 128x128)
#   Trains on a single preset to verify stability before
#   moving to the conditional (multi-class) architecture.
#
# Submit from the final_project/ directory:
#   sbatch jobs/train_dcgan.sh
#
# Prerequisites:
#   mkdir -p logs checkpoints samples
# ============================================================

#SBATCH --job-name=terrain_dcgan
#SBATCH --output=logs/dcgan_%j.out
#SBATCH --error=logs/dcgan_%j.err
#SBATCH --partition=compute          # <-- update to your cluster's CPU partition
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# ---- user-configurable variables ------------------------------------
PRESET=preset_01        # which terrain type to train on
EPOCHS=50
BATCH_SIZE=64
LATENT_DIM=100
LR=0.0002
N_DISC_STEPS=5          # D updates per G update (WGAN-GP standard)
DATASET_DIR=dataset
CHECKPOINT_DIR=checkpoints
SAMPLE_DIR=samples
LOG_DIR=logs
SAMPLE_INTERVAL=5
CHECKPOINT_INTERVAL=10
# ---------------------------------------------------------------------

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Task       : Baseline DCGAN (${PRESET})"
echo "Start time : $(date)"
echo "Node       : $(hostname)"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "========================================"

module load Python
module load TensorFlow     # <-- update module name for your cluster

pip install matplotlib --user 2>&1 | tail -1

# Tell TF/OpenMP to use all allocated cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTEROP_THREADS=1

mkdir -p logs "${CHECKPOINT_DIR}/${PRESET}" "${SAMPLE_DIR}/${PRESET}"

cd "${SLURM_SUBMIT_DIR}" || exit 1

python scripts/train_dcgan.py \
    --preset              "${PRESET}" \
    --epochs              "${EPOCHS}" \
    --batch_size          "${BATCH_SIZE}" \
    --latent_dim          "${LATENT_DIM}" \
    --lr                  "${LR}" \
    --n_disc_steps        "${N_DISC_STEPS}" \
    --dataset_dir         "${DATASET_DIR}" \
    --checkpoint_dir      "${CHECKPOINT_DIR}" \
    --sample_dir          "${SAMPLE_DIR}" \
    --log_dir             "${LOG_DIR}" \
    --sample_interval     "${SAMPLE_INTERVAL}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"

echo "Done: end=$(date)"
