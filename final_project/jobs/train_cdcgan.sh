#!/bin/bash
# ============================================================
# SLURM job — Conditional DCGAN (all 7 terrain presets)
#   Single job; trains one model conditioned on terrain label.
#
# Submit from the final_project/ directory:
#   sbatch jobs/train_cdcgan.sh
#
# Run both jobs concurrently:
#   sbatch jobs/train_dcgan_all.sh && sbatch jobs/train_cdcgan.sh
# ============================================================

#SBATCH --job-name=terrain_cdcgan
#SBATCH --output=logs/cdcgan_%j.out
#SBATCH --error=logs/cdcgan_%j.err
#SBATCH --partition=compute          # <-- update to your cluster's partition
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G                    # full dataset (~600 MB) + TF overhead

# ---- user-configurable variables ------------------------------------
EPOCHS=100
BATCH_SIZE=64
LATENT_DIM=100
LR=0.0002
N_DISC_STEPS=5
DATASET_DIR=dataset
CHECKPOINT_DIR=checkpoints
SAMPLE_DIR=samples
LOG_DIR=logs
SAMPLE_INTERVAL=5
CHECKPOINT_INTERVAL=10
# ---------------------------------------------------------------------

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Task       : Conditional DCGAN (all presets)"
echo "Start time : $(date)"
echo "Node       : $(hostname)"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "========================================"

module load Python
module load TensorFlow     # <-- update module name for your cluster

pip install matplotlib --user 2>&1 | tail -1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_NUM_INTEROP_THREADS=1

mkdir -p logs "${CHECKPOINT_DIR}/cdcgan" "${SAMPLE_DIR}/cdcgan"

cd "${SLURM_SUBMIT_DIR}" || exit 1

python scripts/train_cdcgan.py \
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
