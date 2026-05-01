#!/bin/bash
# ============================================================
# SLURM array job — Train DCNN on Tiny ImageNet-200
#   Array indices 0-2 map to: relu / gelu / rswish
#
# Submit from the assignment2/ directory:
#   sbatch jobs/train.sh
#
# Prerequisites:
#   mkdir -p logs checkpoints results
# ============================================================

#SBATCH --job-name=nn_a2_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --partition=compute          # <-- update to your cluster's CPU partition
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=228G
#SBATCH --array=0-3                # 0-2: compactnet activations; 3: resnet18 baseline

# ---- user-configurable variables ------------------------------------
ACTIVATIONS=(relu gelu rswish resnet18)
DATA_DIR=~/neural-networks/assignment2/data/tiny-imagenet-200   # networked home location
EPOCHS=100
BATCH_SIZE=64
LR=1e-3
WEIGHT_DECAY=1e-4
SEED=42
# ---------------------------------------------------------------------

ACT=${ACTIVATIONS[$SLURM_ARRAY_TASK_ID]}

# Index 3 is the ResNet-18 baseline (activation flag is unused but required)
if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    ARCH=resnet18
    ACT_ARG=relu          # ignored by build_model for resnet18
else
    ARCH=compactnet
    ACT_ARG=${ACT}
fi

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Arch       : ${ARCH}"
echo "Activation : ${ACT_ARG}"
echo "Start time : $(date)"
echo "========================================"

module load Python
module load SciPy-bundle

# Copy dataset from networked home to local scratch for faster I/O
# $TMPDIR is the per-job local scratch directory provided by SLURM
SCRATCH_DATA_DIR="${TMPDIR}/tiny-imagenet-200"
echo "Copying dataset to local scratch: ${SCRATCH_DATA_DIR}"
cp -r "${DATA_DIR}" "${TMPDIR}/"
echo "Copy complete: $(date)"

# Run from assignment2/ regardless of where sbatch was invoked
cd "${SLURM_SUBMIT_DIR}" || exit 1

python train.py \
    --data_dir    "${SCRATCH_DATA_DIR}" \
    --arch        "${ARCH}" \
    --activation  "${ACT_ARG}" \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --lr          "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --seed        "${SEED}" \
    --num_workers 48 \
    --checkpoint_dir checkpoints \
    --results_dir    results

echo "Done: arch=${ARCH}  activation=${ACT_ARG}  end=$(date)"
