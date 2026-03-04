#!/bin/bash
# ============================================================
# SLURM array job — Fine-tune or train from scratch on CIFAR-100
#   Array index 0 = scratch, index 1 = finetune
#
# Run AFTER train.sh completes.  Inspect results/metrics_train_*.json
# to identify the best activation, then set BEST_ACT below.
#
# Submit from the assignment2/ directory:
#   sbatch jobs/finetune.sh
#
# Prerequisites:
#   mkdir -p logs checkpoints results
# ============================================================

#SBATCH --job-name=nn_a2_finetune
#SBATCH --output=logs/finetune_%A_%a.out
#SBATCH --error=logs/finetune_%A_%a.err
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --array=0-1                # 0 = scratch, 1 = finetune

# ---- user-configurable variables ------------------------------------
MODES=(scratch finetune)
BEST_ACT=relu                      # <-- update after reviewing results/metrics_train_*.json
EPOCHS=50
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=1e-4
SEED=42
# ---------------------------------------------------------------------

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
PRETRAINED=checkpoints/best_${BEST_ACT}.pth

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Mode       : ${MODE}"
echo "Activation : ${BEST_ACT}"
echo "Start time : $(date)"
echo "========================================"

module load Python
module load SciPy-Bundle
source /neural-networks/bin/activate

cd "$(dirname "$0")/.." || exit 1

python finetune.py \
    --mode           "${MODE}" \
    --activation     "${BEST_ACT}" \
    --pretrained_path "${PRETRAINED}" \
    --epochs         "${EPOCHS}" \
    --batch_size     "${BATCH_SIZE}" \
    --lr             "${LR}" \
    --weight_decay   "${WEIGHT_DECAY}" \
    --seed           "${SEED}" \
    --num_workers    48 \
    --checkpoint_dir checkpoints \
    --results_dir    results

echo "Done: mode=${MODE}  end=$(date)"
