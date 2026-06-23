#!/bin/bash
# Merges all per-task CSVs and prints the ranked leaderboard.
# Usage: bash aggregate_results.sh [results_dir]

RESULTS_DIR="${1:-results}"

echo "========================================"
echo " Hyperparameter Sweep — Final Rankings"
echo "========================================"
echo ""

python3 - <<EOF
import os, csv, glob

results_dir = "$RESULTS_DIR"
pattern     = os.path.join(results_dir, "result_*.csv")
files       = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].split(".")[0]))

if not files:
    print(f"No result files found in {results_dir}/")
    exit(1)

rows = []
for f in files:
    with open(f) as fh:
        reader = csv.DictReader(fh)
        rows.extend(list(reader))

# Sort by best validation loss (ascending)
rows.sort(key=lambda r: float(r["best_val_loss"]))

header = f"{'Rank':>4} {'Task':>4} {'LR':>8} {'Batch':>5} {'Hidden':>6} {'Dropout':>7} {'Epochs':>6} {'ValLoss':>9} {'ValAcc':>7}"
print(header)
print("-" * len(header))

for i, r in enumerate(rows, 1):
    print(f"{i:>4} {r['task_id']:>4} {r['lr']:>8} {r['batch']:>5} {r['hidden']:>6} "
          f"{r['dropout']:>7} {r['epochs']:>6} {float(r['best_val_loss']):>9.6f} {float(r['val_acc']):>7.4f}")

best = rows[0]
print(f"\nBest config (task {best['task_id']}):")
print(f"  lr={best['lr']}  batch={best['batch']}  hidden={best['hidden']}  "
      f"dropout={best['dropout']}  epochs={best['epochs']}")
print(f"  val_loss={float(best['best_val_loss']):.6f}  val_acc={float(best['val_acc']):.4f}")
print(f"\n{len(rows)} / 12 tasks completed.")
EOF
