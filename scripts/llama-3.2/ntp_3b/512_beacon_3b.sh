#!/bin/bash

# Training script for Activation Beacon with GistDataProcessor
# This script demonstrates how to train Activation Beacon models using
# the same preprocessing pipeline as your own models.

# Get the directory where the script is located and cd to repo root
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Wandb (optional)
# Set WANDB_API_KEY in your environment or run `wandb login` before launching.
export WANDB_PROJECT="ComprExIT"
export WANDB_WATCH="all"

# ===== Multi-node Setup (torchrun + NCCL, NVIDIA GPU) =====
# NOTE: For multi-node training with torchrun (c10d), this script is typically run on EACH node.
# Set NODE_RANK=0..(nnodes-1) per node and keep MASTER_ADDR/MASTER_PORT consistent.
nnodes=1
nproc_per_node=4
node_rank="${NODE_RANK:-0}"

# NETWORK CONFIGURATION
# Priority:
# 1) MASTER_ADDR env var (if already exported by caller)
# 2) Fallback to current hostname
master_addr="${MASTER_ADDR:-}"
if [[ -z "${master_addr}" ]]; then
  master_addr="$(hostname -s 2>/dev/null || hostname)"
fi
export MASTER_ADDR="${master_addr}"
export MASTER_PORT=29501

# ===== Training Configuration =====
run_name="beacon-512-3b-gpu"
output_dir="${OUTPUT_DIR:-./training_outputs}/$run_name"
# --- Required: base model (local path or HF repo id), passed as the first argument ---
model_name_or_path="${1:?Usage: pass <base_model> as the first arg (local path or HF repo id, e.g. meta-llama/Llama-3.2-3B)}"
dataset_folder=${DATA_DIR:-/path/to/datasets}/slim_pajama_6b

# ===== Beacon Configuration =====
enable_beacon=True
beacon_window=64
beacon_stride=64
beacon_attn="full-coverage"
beacon_attend_prev=True
beacon_sink_size=0
beacon_ratio=4
eval_beacon_ratio=$beacon_ratio
beacon_ratio_mix="step-random"
beacon_param="q k v"
beacon_pos="interleave"
group_by_stride="strict"

# ===== Training Hyperparameters =====
max_length=512
context_length=256
generation_length=256
ntp_ratio=1.0
streaming=True
max_tokens="1000 50 50"

per_device_train_batch_size=16
gradient_accumulation_steps=32
max_grad_norm=20.0  
learning_rate=1e-4
num_train_epochs=1
bf16=True

## Logging and evaluation args
eval_strategy="steps"
eval_steps=200
per_device_eval_batch_size=64
logging_steps=10
## Saving
save_strategy="steps"
save_steps=400

gradient_checkpointing=False
use_reentrant=False

ddp_backend="nccl"
ddp_find_unused_parameters=False
seed=42

# ---- Torchrun command (NVIDIA GPU) ----
ulimit -n 65535

uv run torchrun \
    --nnodes="$nnodes" \
    --nproc_per_node="$nproc_per_node" \
    --node_rank="$node_rank" \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    src/baselines/activation_beacon/main/train_with_gist_preprocessing.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset_folder "$dataset_folder" \
    --output_dir "$output_dir" \
    --run_name "$run_name" \
    --enable_beacon "$enable_beacon" \
    --beacon_window "$beacon_window" \
    --beacon_stride "$beacon_stride" \
    --beacon_attn "$beacon_attn" \
    --beacon_attend_prev "$beacon_attend_prev" \
    --beacon_sink_size "$beacon_sink_size" \
    --beacon_ratio "$beacon_ratio" \
    --eval_beacon_ratio "$eval_beacon_ratio" \
    --beacon_ratio_mix "$beacon_ratio_mix" \
    --beacon_param "$beacon_param" \
    --beacon_pos "$beacon_pos" \
    --max_length "$max_length" \
    --context_length "$context_length" \
    --generation_length "$generation_length" \
    --ntp_ratio "$ntp_ratio" \
    --streaming "$streaming" \
    --max_tokens $max_tokens \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$learning_rate" \
    --num_train_epochs "$num_train_epochs" \
    --save_steps "$save_steps" \
    --logging_steps "$logging_steps" \
    --bf16 "$bf16" \
    --do_train \
    --do_eval \
    --gradient_checkpointing "$gradient_checkpointing" \
    --use_reentrant "$use_reentrant" \
    --ddp_backend "$ddp_backend" \
    --ddp_find_unused_parameters "$ddp_find_unused_parameters" \
    --group_by_stride "$group_by_stride" \
    --max_grad_norm "$max_grad_norm" \
    --save_strategy "$save_strategy" \
    --eval_strategy "$eval_strategy" \
    --eval_steps "$eval_steps" \
    --seed "$seed" \
    --report_to wandb \
    --per_device_eval_batch_size "$per_device_eval_batch_size"

# # Example 2: Single Node training
#  python main/train_with_gist_preprocessing.py \
#     --model_name_or_path $model_name_or_path \
#     --dataset_folder $dataset_folder \
#     --output_dir $output_dir \
#     --run_name $run_name \
#     --enable_beacon $enable_beacon \
#     --beacon_window $beacon_window \
#     --beacon_stride $beacon_stride \
#     --beacon_attn $beacon_attn \
#     --beacon_attend_prev $beacon_attend_prev \
#     --beacon_sink_size $beacon_sink_size \
#     --beacon_ratio $beacon_ratio \
#     --beacon_ratio_mix $beacon_ratio_mix \
#     --beacon_param $beacon_param \
#     --beacon_pos $beacon_pos \
#     --max_length $max_length \
#     --context_length $context_length \
#     --generation_length $generation_length \
#     --do_train \
#     --ntp_ratio $ntp_ratio \
#     --streaming $streaming \
#     --max_tokens $max_tokens \
#     --per_device_train_batch_size $per_device_train_batch_size \
#     --gradient_accumulation_steps $gradient_accumulation_steps \
#     --learning_rate $learning_rate \
#     --num_train_epochs $num_train_epochs \
#     --save_steps $save_steps \
#     --logging_steps $logging_steps \
#     --bf16 $bf16 \
#     --gradient_checkpointing $gradient_checkpointing \
#     --use_reentrant $use_reentrant \
#     --ddp_find_unused_parameters $ddp_find_unused_parameters \
#     --group_by_stride $group_by_stride \
#     --save_strategy $save_strategy \
#     --eval_strategy $eval_strategy \
#     --seed $seed \
#     --max_grad_norm $max_grad_norm