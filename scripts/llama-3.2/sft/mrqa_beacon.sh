#!/bin/bash

# Training script for Activation Beacon with GistDataProcessor
# This script demonstrates how to train Activation Beacon models using
# the same preprocessing pipeline as your own models.

# Move to the repo root so relative paths (train.py, etc.) resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Wandb (optional)
# Set WANDB_API_KEY in your environment or run `wandb login` before launching.
export WANDB_PROJECT="ComprExIT"
export WANDB_WATCH="all"

# ===== Multi-node Setup =====
nnodes=1
nproc_per_node=1

# NETWORK CONFIGURATION
# Priority:
# 1) First CLI arg (passed from sbatch wrapper)
# 2) Existing MASTER_ADDR env var (if already exported by caller)
# 3) Fallback to current hostname
master_addr="${MASTER_ADDR:-}"
if [[ -z "${master_addr}" ]]; then
  master_addr="$(hostname -s 2>/dev/null || hostname)"
fi
export MASTER_ADDR="${master_addr}"
export MASTER_PORT=29500

# ===== Training Configuration =====
mode="sft"
# --- Required: directory of the pretrained NTP checkpoint to fine-tune ---
# Pass it as the first argument to this script, e.g.:
#     bash mrqa_beacon.sh "$OUTPUT_DIR/beacon-llama-3.2-1b-256-256-gpu/checkpoint-925"
# (required — there is no environment-variable fallback).
ntp_ckpt="${1:-}"
if [[ -z "${ntp_ckpt}" ]]; then
  echo "Usage: bash $(basename "$0") <ntp_checkpoint_dir>"
  exit 1
fi
model_name_or_path="${ntp_ckpt}"
# Auto-generate like: f"sft_{model_name_or_path.parent.name}"
_model_parent_name="$(basename "$(dirname "${model_name_or_path%/}")")"
run_name="mrqa_${mode}_${_model_parent_name}"
output_dir="${OUTPUT_DIR:-./training_outputs}/$run_name"

# For --mode sft, train_with_gist_preprocessing.py expects datasets under this folder,
# and you must provide --sft_dataset_names (e.g. "squad hotpot_qa").
dataset_folder=${DATA_DIR:-/path/to/datasets}

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
max_length=640
context_length=512
generation_length=128
streaming=False

per_device_train_batch_size=16
gradient_accumulation_steps=32

max_grad_norm=20.0  
learning_rate=1e-4
num_train_epochs=1
bf16=True

## Logging and evaluation args
eval_strategy="steps"
eval_steps=1200
per_device_eval_batch_size=128
logging_steps=10
## Saving
save_strategy="steps"
save_steps=300

gradient_checkpointing=False
use_reentrant=False

# GPU DDP backend
ddp_backend="nccl"
ddp_find_unused_parameters=False
seed=42

# SFT dataset selection (space-separated list)
sft_dataset_names="mrqa_squad mrqa_hotpot_qa mrqa_natural_questions mrqa_trivia_qa mrqa_news_qa mrqa_search_qa"

ulimit -n 65535

# NOTE: For multi-node training with torchrun (c10d), this script usually needs to be run on EACH node.
# Ensure environment variables are consistent across nodes.
uv run torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/baselines/activation_beacon/main/train_with_gist_preprocessing.py \
    --model_name_or_path $model_name_or_path \
    --dataset_folder $dataset_folder \
    --output_dir $output_dir \
    --run_name $run_name \
    --mode $mode \
    --sft_dataset_names $sft_dataset_names \
    --enable_beacon $enable_beacon \
    --beacon_window $beacon_window \
    --beacon_stride $beacon_stride \
    --beacon_attn $beacon_attn \
    --beacon_attend_prev $beacon_attend_prev \
    --beacon_sink_size $beacon_sink_size \
    --beacon_ratio $beacon_ratio \
    --eval_beacon_ratio $eval_beacon_ratio \
    --beacon_ratio_mix $beacon_ratio_mix \
    --beacon_param $beacon_param \
    --beacon_pos $beacon_pos \
    --max_length $max_length \
    --context_length $context_length \
    --generation_length $generation_length \
    --streaming $streaming \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --save_steps $save_steps \
    --logging_steps $logging_steps \
    --bf16 $bf16 \
    --do_train \
    --gradient_checkpointing $gradient_checkpointing \
    --use_reentrant $use_reentrant \
    --ddp_backend $ddp_backend \
    --ddp_find_unused_parameters $ddp_find_unused_parameters \
    --group_by_stride $group_by_stride \
    --max_grad_norm $max_grad_norm \
    --save_strategy $save_strategy \
    --eval_strategy $eval_strategy \
    --eval_steps $eval_steps \
    --seed $seed \
    --report_to wandb \
    --per_device_eval_batch_size $per_device_eval_batch_size