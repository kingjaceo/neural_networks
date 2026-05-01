#!/bin/bash
# ============================================================
# SLURM job — Train and evaluate CIFAR-100 ensemble (Task 2)
#
# Run AFTER finetune.sh completes.  Set BEST_ACT to the
# activation with the best results/metrics_finetune_*.json.
#
# Submit from the assignment2/ directory:
#   sbatch jobs/ensemble.sh
#
# Prerequisites:
#   mkdir -p logs checkpoints results
# ============================================================

#SBATCH --job-name=nn_a2_ensemble
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err
#SBATCH --partition=compute          # <-- update to your cluster's CPU partition
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=228G

# ---- user-configurable variables ------------------------------------
BEST_ACT=rswish                    # <-- update after reviewing results/metrics_finetune_*.json
EPOCHS=50
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=1e-4
SEEDS="42 43 44"
# ---------------------------------------------------------------------

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Activation : ${BEST_ACT}"
echo "Start time : $(date)"
echo "========================================"

module load Python
module load SciPy-bundle

cd "${SLURM_SUBMIT_DIR}" || exit 1

python ensemble.py \
    --activation     "${BEST_ACT}" \
    --pretrained_dir checkpoints \
    --epochs         "${EPOCHS}" \
    --batch_size     "${BATCH_SIZE}" \
    --lr             "${LR}" \
    --weight_decay   "${WEIGHT_DECAY}" \
    --seeds          ${SEEDS} \
    --num_workers    48 \
    --checkpoint_dir checkpoints \
    --results_dir    results

echo "Done: activation=${BEST_ACT}  end=$(date)"
