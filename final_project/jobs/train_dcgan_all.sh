#!/bin/bash
# ============================================================
# SLURM job array — Baseline Unconditional DCGAN (all presets)
#   Submits one task per preset, all running in parallel.
#
# Submit from the final_project/ directory:
#   sbatch jobs/train_dcgan_all.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/dcgan_array_<jobid>_<taskid>.out
# ============================================================

#SBATCH --job-name=terrain_dcgan
#SBATCH --output=logs/dcgan_array_%A_%a.out
#SBATCH --error=logs/dcgan_array_%A_%a.err
#SBATCH --array=1-7                          # one task per preset
#SBATCH --partition=compute                  # <-- update to your cluster's partition
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# ---- user-configurable variables ------------------------------------
EPOCHS=50
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

PRESET="preset_$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"

echo "========================================"
echo "Job ID     : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Task       : Baseline DCGAN (${PRESET})"
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

echo "Done: preset=${PRESET} end=$(date)"
