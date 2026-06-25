#!/bin/bash
#SBATCH --job-name=hp_sweep
#SBATCH --output=results/task_%A_%a.out
#SBATCH --error=results/task_%A_%a.err
#SBATCH --array=1-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:15:00
#SBATCH --partition=normal

# Each array task picks one row from the CSV (skip the header row)
PARAM_FILE="${SLURM_SUBMIT_DIR}/configs/sweep_params.csv"
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_FILE")

LR=$(echo "$ROW"         | cut -d, -f1)
BATCH=$(echo "$ROW"      | cut -d, -f2)
HIDDEN=$(echo "$ROW"     | cut -d, -f3)
DROPOUT=$(echo "$ROW"    | cut -d, -f4)
EPOCHS=$(echo "$ROW"     | cut -d, -f5)

echo "============================================"
echo "Task    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "Job ID  : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node    : $(hostname)"
echo "Config  : lr=$LR  batch=$BATCH  hidden=$HIDDEN  dropout=$DROPOUT  epochs=$EPOCHS"
echo "============================================"
echo ""

# Simulate training (SGD with noise - reproducible per task)
python3 - <<EOF
import math, random

random.seed($SLURM_ARRAY_TASK_ID)

lr      = float("$LR")
batch   = int("$BATCH")
hidden  = int("$HIDDEN")
dropout = float("$DROPOUT")
epochs  = int("$EPOCHS")

loss     = 2.5 + random.gauss(0, 0.3)
best_val = float("inf")
history  = []

for epoch in range(1, epochs + 1):
    # SGD step with momentum + noise
    grad_noise = random.gauss(0, 0.05)
    loss -= lr * (loss - 0.05) * (1 - grad_noise) * (batch / 64.0) ** 0.5
    loss = max(loss, 0.05 + random.uniform(0, 0.02))

    val_loss = loss * (1 + dropout * random.gauss(0, 0.3))
    val_loss = max(val_loss, 0.04)
    if val_loss < best_val:
        best_val = val_loss

    if epoch % max(1, epochs // 10) == 0:
        print(f"  Epoch {epoch:4d}/{epochs} | train={loss:.4f} | val={val_loss:.4f}")

val_acc = max(0.0, 1.0 - best_val * 0.8 + random.gauss(0, 0.01))
val_acc = min(val_acc, 0.999)

print(f"\n--- Final Results (task $SLURM_ARRAY_TASK_ID) ---")
print(f"Config    : lr={lr}  batch={batch}  hidden={hidden}  dropout={dropout}")
print(f"Best val  : {best_val:.6f}")
print(f"Val acc   : {val_acc:.4f}")

# Write result for aggregation
import os
out_dir = os.path.join(os.environ.get("SLURM_SUBMIT_DIR", "."), "results")
with open(f"{out_dir}/result_$SLURM_ARRAY_TASK_ID.csv", "w") as f:
    f.write("task_id,lr,batch,hidden,dropout,epochs,best_val_loss,val_acc\n")
    f.write(f"$SLURM_ARRAY_TASK_ID,{lr},{batch},{hidden},{dropout},{epochs},{best_val:.6f},{val_acc:.4f}\n")
EOF

echo ""
echo "Result written to results/result_${SLURM_ARRAY_TASK_ID}.csv"
