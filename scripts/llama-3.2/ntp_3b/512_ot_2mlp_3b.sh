# Move to the repo root so relative paths (train.py, etc.) resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Wandb
# Set WANDB_API_KEY in your environment or run `wandb login` before launching.
export WANDB_PROJECT="ComprExIT"
export WANDB_WATCH="all"

##### wandb run name ####
run_name="512-128-3b-ot-learnable-a-ntp1.0-gpu"

# Multi-node setup
model_structure="hier"
nnodes=1
nproc_per_node=4

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

# General control args
output_dir="${OUTPUT_DIR:-./training_outputs}/$run_name"
# --- Required: base model (local path or HF repo id), passed as the first argument ---
model_name_or_path="${1:?Usage: pass <base_model> as the first arg (local path or HF repo id, e.g. meta-llama/Llama-3.2-3B)}"
dataset_folder=${DATA_DIR:-/path/to/datasets}/slim_pajama_6b
## Logging and evaluation args
eval_strategy="steps"
eval_steps=200
logging_steps=10
## Saving
save_strategy="steps"
save_steps=400

# debug args (comment out when doing formal training)
max_tokens="1000 50 50" # max number of tokens for each split in million
num_samples=-1 # for debugging use
max_train_samples=-1
max_eval_samples=-1
overwrite_cache=True
shuffle_train_set=True

# model structure
add_global_avg=False 
## LoRA
lora_r=128
lora_alpha=32
lora_dropout=0.05
lora_bias="none"
lora_task_type="CAUSAL_LM"

# -----

# training hyper-parameters
streaming=True 
num_train_epochs=1
warmup_ratio=0.05

# Mixed precision training
# Use bf16 for GPU - more stable than fp16
dtype=bfloat16  # dtype for loading frozen components
bf16=True  # Enable bf16 mixed precision training in Trainer

gradient_checkpointing=False

ntp_ratio=1.0
max_grad_norm=20.0
lora_compressor=False
learning_rate=1e-4

per_device_train_batch_size=16
gradient_accumulation_steps=32

num_hidden_layers=2 # @@CRITICAL ARG@@
top_k_layers=-1 # @@CRITICAL ARG@@
pooling_method="ot-dy-src" # @@CRITICAL ARG@@

training_freezing_mode=compress+llm  # Options: compress+llm, both, compress, llm, projector, none

# For pooling
compression_ratio=4 # @@CRITICAL ARG@@
layerwise_pooling_gate_hidden=256
layerwise_pooling_temperature=0.7
# --- For OptimalTransportPooling ("ot") ---
ot_window_size=128
ot_n_iter=30
ot_metric_dim=$layerwise_pooling_gate_hidden

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
    --top_k_layers $top_k_layers \
    --pooling_method $pooling_method \
    --training_freezing_mode $training_freezing_mode \
    --dtype $dtype \
    --bf16 $bf16 \
    --num_samples $num_samples \
    --max_train_samples $max_train_samples \
    --max_eval_samples $max_eval_samples \
    --overwrite_cache $overwrite_cache \
    --max_tokens $max_tokens \
    --ddp_backend nccl \
    --save_strategy $save_strategy \
    --eval_strategy $eval_strategy \
    --eval_steps $eval_steps \
    --ddp_find_unused_parameters False \
    --max_grad_norm $max_grad_norm \
    --ntp_ratio $ntp_ratio \
    --num_hidden_layers $num_hidden_layers \
    --add_global_avg $add_global_avg \
    --shuffle_train_set $shuffle_train_set \
    --lora_compressor $lora_compressor \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --lora_bias $lora_bias \
    --compression_ratio $compression_ratio \
    --lora_task_type $lora_task_type \
    --layerwise_pooling_gate_hidden $layerwise_pooling_gate_hidden \
    --layerwise_pooling_temperature $layerwise_pooling_temperature \
    --ot_window_size $ot_window_size \
    --ot_n_iter $ot_n_iter \
    --ot_metric_dim $ot_metric_dim \
    --gradient_checkpointing $gradient_checkpointing
