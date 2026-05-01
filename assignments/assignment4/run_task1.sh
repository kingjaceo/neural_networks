#!/bin/bash
# ============================================================
# SLURM job — Task 1: Deep Convolutional Autoencoder (CIFAR-10)
#   Trains a DCAE and evaluates latent-space quality via t-SNE.
#
# Submit from the assignment4/ directory:
#   sbatch run_task1.sh
#
# Prerequisites:
#   mkdir -p slurm_logs outputs
# ============================================================

#SBATCH --job-name=nn_a4_task1
#SBATCH --output=slurm_logs/task1_%j.out
#SBATCH --error=slurm_logs/task1_%j.err
#SBATCH --partition=compute              # <-- update to your cluster's CPU partition
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=228G

# ---- user-configurable variables ------------------------------------
EPOCHS=50
LATENT_DIM=512
BATCH_SIZE=128
LR=1e-3
DATA_DIR=./data
SEED=42
TSNE_SAMPLES=10000
NUM_WORKERS=48
# ---------------------------------------------------------------------

OUTDIR="outputs/task1_${SLURM_JOB_ID}"
mkdir -p slurm_logs "${OUTDIR}"

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Task       : 1 — DCAE CIFAR-10"
echo "Node       : ${SLURMD_NODENAME}"
echo "Output     : ${OUTDIR}"
echo "Start time : $(date)"
echo "========================================"

module load Python
module load SciPy-bundle
module load 
pip install scikit-image --user 2>&1 | tail -1


cd "${SLURM_SUBMIT_DIR}" || exit 1

python task1_dcae_cifar10.py \
    --epochs         "${EPOCHS}" \
    --latent-dim     "${LATENT_DIM}" \
    --batch-size     "${BATCH_SIZE}" \
    --lr             "${LR}" \
    --data-dir       "${DATA_DIR}" \
    --out-dir        "${OUTDIR}" \
    --num-workers    "${NUM_WORKERS}" \
    --seed           "${SEED}" \
    --tsne-samples   "${TSNE_SAMPLES}"

echo "Done: Task 1  end=$(date)"
