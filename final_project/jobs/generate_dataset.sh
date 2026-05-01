#!/bin/bash
# ============================================================
# SLURM job — Terrain Dataset Generation
#   Generates N_TRAIN × n_presets training images and
#   N_TEST × n_presets test images, saved as .npy arrays.
#
# Submit from the final_project/ directory:
#   sbatch jobs/generate_dataset.sh
#
# Prerequisites:
#   mkdir -p logs dataset
#   Edit SELECTED_PRESETS in generate_dataset.py first
#   Run preview_presets.py locally to pick presets
# ============================================================

#SBATCH --job-name=terrain_datagen
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err
#SBATCH --partition=compute          # <-- update to your cluster's CPU partition
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G

# ---- user-configurable variables ------------------------------------
OUTPUT_DIR=dataset
N_TRAIN=3000       # per preset  (7 presets × 3000 = 21000 train)
N_TEST=200         # per preset  (7 presets × 200  = 1400 test)
SEED=42
# ---------------------------------------------------------------------

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Task       : Terrain Dataset Generation"
echo "Start time : $(date)"
echo "Node       : $(hostname)"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "========================================"

module load Python
module load SciPy-bundle

pip install opensimplex pillow --user 2>&1 | tail -2

mkdir -p logs "${OUTPUT_DIR}"

cd "${SLURM_SUBMIT_DIR}" || exit 1

python scripts/generate_dataset.py \
    --output_dir "${OUTPUT_DIR}" \
    --n_train    "${N_TRAIN}"    \
    --n_test     "${N_TEST}"     \
    --seed       "${SEED}"       \
    --workers    "${SLURM_CPUS_PER_TASK}"

echo "Done: end=$(date)"
