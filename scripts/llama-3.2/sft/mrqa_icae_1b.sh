# Move to the repo root so relative paths (train.py, etc.) resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Wandb
# Set WANDB_API_KEY in your environment or run `wandb login` before launching.
export WANDB_PROJECT="ComprExIT"
export WANDB_WATCH="all"

##### wandb run name ####
mode="sft"
# --- Required: directory of the pretrained NTP checkpoint to fine-tune ---
# Pass it as the first argument to this script, e.g.:
#     bash mrqa_icae_1b.sh "$OUTPUT_DIR/512-128-icae-gpu/checkpoint-763"
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

# Multi-node setup
model_structure="hier"
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
export MASTER_PORT=29503

# General control args
output_dir="${OUTPUT_DIR:-./training_outputs}/$run_name"
dataset_folder=${DATA_DIR:-/path/to/datasets}
## Logging and evaluation args
eval_strategy="steps"
eval_steps=600
logging_steps=5
## Saving
save_strategy="steps"
save_steps=200

# debug args (comment out when doing formal training)
num_samples=-1 # for debugging use
max_train_samples=-1
max_eval_samples=-1
overwrite_cache=False

# model structure
## LoRA
lora_r=128
lora_alpha=32
lora_dropout=0.05
lora_bias="none"
lora_task_type="CAUSAL_LM"

# -----

# training hyper-parameters
streaming=False 
num_train_epochs=1
warmup_ratio=0.00
shuffle_train_set=True

# Mixed precision training
# Use bf16 for GPU - more stable than fp16
dtype=bfloat16  # dtype for loading frozen components
bf16=True  # Enable bf16 mixed precision training in Trainer

gradient_checkpointing=False

sft_dataset_names="mrqa_squad mrqa_hotpot_qa mrqa_natural_questions mrqa_trivia_qa mrqa_news_qa mrqa_search_qa"

ntp_ratio=1.0
max_grad_norm=20.0
lora_compressor=False
learning_rate=1e-4

per_device_train_batch_size=32
gradient_accumulation_steps=16
training_freezing_mode=compress+llm  # Options: compress+llm, both, compress, llm, projector, none

# input size
context_length=512 # @@CRITICAL ARG@@
generation_length=128 # @@CRITICAL ARG@@

# Torchrun command
# NOTE: For multi-node training with torchrun (c10d), this script usually needs to be run on EACH node.
# Ensure environment variables are consistent.

ulimit -n 65535

uv run torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --output_dir $output_dir \
    --run_name $run_name \
    --mode $mode \
    --model_structure $model_structure \
    --logging_steps $logging_steps \
    --warmup_ratio $warmup_ratio \
    --save_steps $save_steps \
    --model_name_or_path $model_name_or_path \
    --dataset_folder $dataset_folder \
    --do_train \
    --do_eval \
    --streaming $streaming \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs $num_train_epochs \
    --learning_rate $learning_rate \
    --context_length $context_length \
    --generation_length $generation_length \
    --training_freezing_mode $training_freezing_mode \
    --dtype $dtype \
    --bf16 $bf16 \
    --num_samples $num_samples \
    --max_train_samples $max_train_samples \
    --max_eval_samples $max_eval_samples \
    --overwrite_cache $overwrite_cache \
    --ddp_backend nccl \
    --save_strategy $save_strategy \
    --eval_strategy $eval_strategy \
    --eval_steps $eval_steps \
    --ddp_find_unused_parameters False \
    --max_grad_norm $max_grad_norm \
    --ntp_ratio $ntp_ratio \
    --shuffle_train_set $shuffle_train_set \
    --lora_compressor $lora_compressor \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_bias $lora_bias \
    --lora_task_type $lora_task_type \
    --gradient_checkpointing $gradient_checkpointing \
    --sft_dataset_names $sft_dataset_names \
    --attn_implementation eager