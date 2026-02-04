#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/arc1_task_list.sh"
file_names=("${ARC1_TASK_NAMES[@]}")

LOOP_CKPT="checkpoints/loop_vit_base.pt"

gpu_ids=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#gpu_ids[@]}

for (( gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++ )); do
  gpu=${gpu_ids[$gpu_idx]}
  (
    i=0
    for file_name in "${file_names[@]}"; do
      if (( i % NUM_GPUS == gpu_idx )); then
        echo "GPU ${gpu} processing task ${file_name} (ARC-1 LoopViT)"
        CUDA_VISIBLE_DEVICES=${gpu} python test_time_train_ARC.py \
          --epochs 100 \
          --batch-size 8 \
          --image-size 64 \
          --patch-size 2 \
          --learning-rate 3e-4 \
          --weight-decay 0 \
          --embed-dim 512 \
          --mlp-dim 512 \
          --num-heads 8 \
          --num-colors 12 \
          --loop-core-depth 6 \
          --max-loop-steps 6 \
          --min-loop-steps 6 \
          --resume-checkpoint "${LOOP_CKPT}" \
          --lr-scheduler "cosine" \
          --train-split "eval_color_permute_ttt_9/${file_name}" \
          --data-root "raw_data/ARC-AGI" \
          --eval-split "eval_color_permute_ttt_9/${file_name}" \
          --resume-skip-task-token \
          --architecture "loop_vit" \
          --eval-save-name "ttt_results" \
          --no-compile \
          --num-attempts 10 \
          --ttt-num-each 2
      fi
      ((i++))
    done
  ) &
done

wait
echo "All LoopARCViT ARC-1 tasks finished."
