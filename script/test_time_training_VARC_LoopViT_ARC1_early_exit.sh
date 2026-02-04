SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PARENT_DIR}" || exit

source "${SCRIPT_DIR}/arc1_task_list.sh"
file_names=("${ARC1_TASK_NAMES[@]}")

LOOP_CKPT="checkpoints/loop_vit_base.pt"

# Use all available GPUs
gpu_ids=(0 1 2 3 4 5 6 7) 
NUM_GPUS=${#gpu_ids[@]}

for (( gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++ )); do
  gpu=${gpu_ids[$gpu_idx]}
  (
    i=0
    for file_name in "${file_names[@]}"; do
      if (( i % NUM_GPUS == gpu_idx )); then
        echo "GPU ${gpu} processing task ${file_name} (ARC-1 Early Exit)"
        CUDA_VISIBLE_DEVICES=${gpu} python test_time_train_ARC_vis.py \
          --epochs 100 \
          --batch-size 1 \
          --image-size 64 \
          --patch-size 2 \
          --learning-rate 3e-4 \
          --weight-decay 0 \
          --embed-dim 512 \
          --mlp-dim 512 \
          --num-heads 8 \
          --num-colors 12 \
          --loop-core-depth 2 \
          --max-loop-steps 6 \
          --min-loop-steps 1 \
          --resume-checkpoint "${LOOP_CKPT}" \
          --lr-scheduler "cosine" \
          --train-split "eval_color_permute_ttt_9/${file_name}" \
          --data-root "raw_data/ARC-AGI" \
          --eval-split "eval_color_permute_ttt_9/${file_name}" \
          --resume-skip-task-token \
          --architecture "loop_vit" \
          --eval-save-name "ttt_results_early_exit" \
          --num-attempts 1 \
          --ttt-num-each 1 \
          --visualize-loop-steps \
          --visualize-attention \
          --exit-on-entropy-stable \
          --entropy-patience 2 \
          --entropy-threshold-pct 0.05
      fi
      ((i++))
    done
  ) &
done

wait
echo "All LoopARCViT ARC-1 Early Exit tasks finished."
