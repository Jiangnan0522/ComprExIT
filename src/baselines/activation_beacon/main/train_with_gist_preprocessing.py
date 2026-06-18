import math
import logging
import sys
import os
from datetime import datetime
from functools import partial
import torch
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from datasets import load_dataset, IterableDatasetDict
import datasets
from typing import Dict, List, Any
from dataclasses import dataclass, field
from transformers.tokenization_utils import PreTrainedTokenizer
import traceback
from loguru import logger

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add the CompressIn src directory to import GistDataProcessor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src import ( 
    DefaultDataCollator,
    FileLogger,
    get_model_and_tokenizer,
    makedirs,
    format_numel_str
)
from src.utils import get_max_length_in_nested_lists, pad_nested_lists
from src.args_with_gist import ModelArgs, GistTrainingArgs
from src.metrics import Metric
from src.trainer import ActivationBeaconTrainer

# Import preprocessing + data loading utilities from CompressIn src/data_processing
from data_processing.ntp_preprocessing import GistDataProcessor
from data_processing.sft_preprocessing import SFTDataProcessor
from data_processing.data_loading import data_loading_factory, sft_data_loading_factory
from data_processing.prompt_template import QATemplate

PAD_TOKEN = "<pad>"

@dataclass
class GistCompatibleDataCollator:
    """
    Data collator that extends DefaultDataCollator to handle GistDataProcessor outputs.
    Removes fields that are specific to CompressIn preprocessing but not consumed by activation_beacon models.
    """
    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    keys_to_tensorize = {"input_ids", "attention_mask", "labels", "position_ids", "token_type_ids", "length", "depth", "index"}
    keys_to_remove = {"for_ntp", "reconstruction_segment", "context_mask"}

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        # Remove GistDataProcessor-specific fields
        cleaned_batch = []
        for elem in batch_elem:
            cleaned_elem = {k: v for k, v in elem.items() if k not in self.keys_to_remove}
            cleaned_batch.append(cleaned_elem)
        
        first_elem = cleaned_batch[0]
        return_batch = {}
        
        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in cleaned_batch]
            # pad all lists and nested lists
            if isinstance(value, list) and key in self.keys_to_tensorize:
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length, pad_token_id, self.tokenizer.padding_side)

            if key in self.keys_to_tensorize and None not in batch_value:
                return_batch[key] = torch.tensor(batch_value)
            else:
                # handle strings and None
                return_batch[key] = batch_value
        return return_batch


def main():
    parser = HfArgumentParser([ModelArgs, GistTrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Configure loguru to write to both console and file
    logger.add(sys.stderr, format="{time} {level} {file.name}:{function} - {message}", level="INFO")
    # IMPORTANT: Never have multiple ranks write to the same log file.
    # If you want a single log file, restrict file logging to rank 0 only.
    global_rank = int(
        os.environ.get("RANK")
        or os.environ.get("PMI_RANK")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("MPI_RANKID")
        or "0"
    )
    if global_rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(training_args.output_dir, f"training_{ts}.log")
        logger.add(log_file_path, rotation="10 MB", level="DEBUG")
        logger.info(f"Logging to file: {log_file_path}")
    else:
        logger.info(f"[Global rank {global_rank}] File logging disabled (stderr only)")


    # Determine device (XPU or CUDA)
    if torch.xpu.is_available():
        device = "xpu"
        logger.info(f"[Rank {training_args.local_rank}] Using Intel XPU for training")
        logger.info(f"[Rank {training_args.local_rank}] XPU device count: {torch.xpu.device_count()}")
    else:
        device = "cuda"
        logger.info(f"[Rank {training_args.local_rank}] Using CUDA for training")

    # Log XPU memory before loading model
    if torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            mem_allocated = torch.xpu.memory_allocated(i) / 1024**3
            mem_reserved = torch.xpu.memory_reserved(i) / 1024**3
            logger.info(f"[Rank {training_args.local_rank}] XPU {i} memory before model loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    
    logger.info(f"[Rank {training_args.local_rank}] About to load model from {model_args.model_name_or_path}")
    model, tokenizer = get_model_and_tokenizer(model_args, device=device, evaluation_mode=False)
    logger.info(f"[Rank {training_args.local_rank}] Model loaded successfully")
    
    # Log XPU memory after loading model
    if torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            mem_allocated = torch.xpu.memory_allocated(i) / 1024**3
            mem_reserved = torch.xpu.memory_reserved(i) / 1024**3
            logger.info(f"[Rank {training_args.local_rank}] XPU {i} memory after model loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

    # Log model training mode
    logger.info(f"Model training mode: {model.training}")

    mode = getattr(training_args, "mode", "ntp")
    if mode not in {"ntp", "sft"}:
        raise ValueError(f"Invalid --mode: {mode}. Expected one of: ['ntp', 'sft']")
    
    if model_args.enable_beacon and training_args.only_train_beacon:
        logger.info("Freezing non-beacon parameters (only_train_beacon=True)")
        frozen_count = 0
        for name, param in model.named_parameters():
            if "beacon" not in name:
                param.requires_grad_(False)
                frozen_count += 1
        logger.info(f"Frozen {frozen_count} non-beacon parameters")

    # Log trainable parameters count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    
    # Critical check: ensure we have at least some trainable parameters
    if trainable_params == 0:
        raise ValueError(
            "No trainable parameters found! This will cause gradient errors during training. "
            "If you're using --only_train_beacon True, make sure the model has beacon parameters initialized. "
            "Otherwise, set --only_train_beacon False to train all parameters."
        )
    

    if training_args.lora_tune:
        from peft import (
            LoraConfig,
            get_peft_model,
        )
        # copied from LongLoRA
        config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_targets,
            modules_to_save=training_args.lora_extra_params,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    # ===== Shared lengths (used for both ntp and sft) =====
    # Calculate context_length and generation_length from max_length or use provided values.
    if training_args.context_length is not None:
        context_length = training_args.context_length
    else:
        # Default: split the sequence equally
        context_length = model_args.max_length // 2
    
    if training_args.generation_length is not None:
        generation_length = training_args.generation_length
    else:
        generation_length = model_args.max_length - context_length
    
    ntp_ratio = training_args.ntp_ratio

    # ===== Load datasets =====
    train_dataset = None
    eval_dataset = None
    
    logger.info(f"[Rank {training_args.local_rank}] About to process dataset")
    dist_inited = bool(getattr(torch, "distributed", None)) and torch.distributed.is_available() and torch.distributed.is_initialized()
    logger.info(f"[Rank {training_args.local_rank}] Entering main_process_first (torch.distributed initialized={dist_inited})")
    
    with training_args.main_process_first():
        logger.info(f"[Rank {training_args.local_rank}] Main process: processing dataset")
        if mode == "ntp":
            # ===== NTP (plain text) preprocessing via GistDataProcessor =====
            data_processor = GistDataProcessor(
                tokenizer=tokenizer,
                context_length=context_length,
                generation_length=generation_length,
                seed=training_args.seed,
                ntp_ratio=ntp_ratio,
            )
            logger.info(
                f"Initialized GistDataProcessor (mode=ntp) with context_length={context_length}, "
                f"generation_length={generation_length}, ntp_ratio={ntp_ratio}"
            )

            # Method 1: Use dataset_folder (like CompressIn's train.py)
            if training_args.dataset_folder is not None:
                logger.info(f"Loading dataset from folder: {training_args.dataset_folder}")

                # Get dataset name from folder
                dataset_name = os.path.basename(training_args.dataset_folder)
                raw_datasets, columns_to_remove = data_loading_factory(
                    name=dataset_name,
                    data_folder=training_args.dataset_folder,
                    streaming=training_args.streaming,
                )

                # Limit samples if num_samples is set (for debugging)
                if (training_args.num_samples is not None) and (training_args.num_samples > 0):
                    if training_args.streaming:
                        raw_datasets = IterableDatasetDict({
                            k: v.take(training_args.num_samples) for k, v in raw_datasets.items()
                        })
                    else:
                        raw_datasets = datasets.DatasetDict({
                            k: v.select(range(min(len(v), training_args.num_samples)))
                            for k, v in raw_datasets.items()
                        })
                    logger.info(f"num_samples is set to {training_args.num_samples}. Selected {training_args.num_samples} samples for each split.")

                # Process with GistDataProcessor
                logger.info("Processing dataset with GistDataProcessor (mode=ntp)...")
                processed_datasets = data_processor.process_dataset(
                    raw_datasets=raw_datasets,
                    text_column_name='text',
                    columns_to_remove=columns_to_remove,
                    streaming=training_args.streaming,
                    preprocessing_num_workers=training_args.preprocessing_num_workers,
                    overwrite_cache=training_args.overwrite_cache,
                    shuffle_train_set=True,
                    max_tokens=training_args.max_tokens,
                    max_train_samples=training_args.max_train_samples if training_args.max_train_samples and training_args.max_train_samples > 0 else None,
                    max_eval_samples=training_args.max_eval_num if training_args.max_eval_num else None,
                )

                train_dataset = processed_datasets.get('train', None)
                eval_dataset = processed_datasets.get('validation', None)

                logger.info(f"Dataset loaded and processed. Max samples: {data_processor.max_samples}")
            else:
                raise ValueError("--dataset_folder must be provided!")

        else:
            # ===== SFT preprocessing =====
            if training_args.streaming:
                logger.warning("--streaming is not supported for --mode sft. Setting streaming=False.")
                training_args.streaming = False

            # Ensure pad token is defined and distinct from eos for SFT fixed-length packing.
            if (tokenizer.pad_token_id is None) or (tokenizer.pad_token_id == tokenizer.eos_token_id):
                logger.info("Tokenizer pad_token_id is unset or equals eos_token_id. Adding a dedicated <pad> token for SFT.")
                tokenizer.add_special_tokens({"additional_special_tokens": [PAD_TOKEN]})
                tokenizer.pad_token = PAD_TOKEN
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
                # Resize model embeddings if needed
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception as e:
                    logger.warning(f"Failed to resize token embeddings after adding pad token: {e}")

            # Case A: dataset_folder + --sft_dataset_names (QA-style datasets: context/question/gold)
            if training_args.dataset_folder is not None:
                if not training_args.sft_dataset_names:
                    raise ValueError("--mode sft with --dataset_folder requires --sft_dataset_names (e.g. --sft_dataset_names squad)")

                logger.info(
                    f"Loading SFT datasets {training_args.sft_dataset_names} from folder: {training_args.dataset_folder}"
                )
                raw_datasets, _ = sft_data_loading_factory(
                    dataset_names=training_args.sft_dataset_names,
                    datasets_dir=training_args.dataset_folder,
                )

                # Limit samples if num_samples is set (for debugging)
                if (training_args.num_samples is not None) and (training_args.num_samples > 0):
                    raw_datasets = datasets.DatasetDict({
                        k: v.select(range(min(len(v), training_args.num_samples)))
                        for k, v in raw_datasets.items()
                    })
                    logger.info(f"num_samples is set to {training_args.num_samples}. Selected {training_args.num_samples} samples for each split.")

                logger.info("Processing dataset with SFTDataProcessor (QA-style)...")
                sft_processor = SFTDataProcessor(
                    tokenizer=tokenizer,
                    max_context_length=context_length,
                    max_generation_length=generation_length,
                    seed=training_args.seed,
                )
                processed_datasets = sft_processor.process_dataset(
                    dataset=raw_datasets,
                    shuffle_train_set=True,
                    overwrite_cache=training_args.overwrite_cache,
                )

                # Convert query-only labels -> full-length labels aligned to input_ids (required by ActivationBeaconTrainer/model)
                def _to_full_labels(example: Dict[str, Any]) -> Dict[str, Any]:
                    return {"labels": ([-100] * context_length) + example["labels"]}

                fixed = {}
                for split, ds in processed_datasets.items():
                    remove_cols = ["context_mask"] if "context_mask" in ds.column_names else []
                    fixed[split] = ds.map(
                        _to_full_labels,
                        remove_columns=remove_cols,
                        desc=f"Postprocess SFT labels for activation_beacon ({split})",
                    )
                processed_datasets = datasets.DatasetDict(fixed)

                train_dataset = processed_datasets.get("train", None)

                # Eval path: support both perplexity and generation-style evaluation
                if training_args.eval_method == "generation" and "validation" in raw_datasets:
                    template = QATemplate()

                    def _build_eval_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
                        # Build: [context] + [Question...Answer: ] (no gold tokens)
                        context_text = template.build_context_text(example["context"])
                        prefix_text = template.build_prefix_text(example["question"])

                        context_ids = tokenizer.encode(context_text, add_special_tokens=False)
                        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

                        # Keep within model max_length by truncating context first (left truncation)
                        max_prompt_len = int(model_args.max_length)
                        if len(prefix_ids) >= max_prompt_len:
                            prefix_ids = prefix_ids[-max_prompt_len:]
                            context_ids = []
                        else:
                            budget = max_prompt_len - len(prefix_ids)
                            if len(context_ids) > budget:
                                context_ids = context_ids[-budget:]

                        input_ids = context_ids + prefix_ids
                        attention_mask = [1] * len(input_ids)
                        return {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            # Generation eval expects string labels
                            "labels": example["gold"],
                            "length": len(input_ids),
                        }

                    eval_dataset = raw_datasets["validation"].map(
                        _build_eval_prompt,
                        remove_columns=list(raw_datasets["validation"].column_names),
                        desc="Build generation prompts for SFT validation (activation_beacon)",
                    )
                else:
                    eval_dataset = processed_datasets.get("validation", None)

                logger.info(
                    f"SFT datasets processed. Train split: {train_dataset is not None}, "
                    f"Validation split: {eval_dataset is not None}"
                )

            else:
                raise ValueError("For --mode sft, please provide --dataset_folder + --sft_dataset_names")
    
    logger.info(f"[Rank {training_args.local_rank}] Exited main_process_first; continuing setup")

    # Use the custom data collator that handles GistDataProcessor outputs
    # This collator removes the 'for_ntp' and 'reconstruction_segment' fields
    # that are specific to GistDataProcessor but not needed by activation_beacon
    data_collator = GistCompatibleDataCollator(tokenizer)

    # Calculate the number of steps
    if (training_args.max_steps is None or training_args.max_steps < 0) and training_args.streaming and mode == "ntp":
        actual_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        max_steps_per_epoch = math.ceil(data_processor.max_samples / actual_batch_size)
        max_steps = math.ceil(max_steps_per_epoch * training_args.num_train_epochs )   
        training_args.max_steps = max_steps
        logger.info(f"[Max steps] Calculated max steps in each epoch: {max_steps_per_epoch}")
        logger.info(f"[Max steps] Calculated max steps in total: {max_steps}")

    trainer = ActivationBeaconTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        file_logger=FileLogger(makedirs(training_args.log_path)),
        compute_metrics=Metric.get_metric_fn(
            metrics=training_args.metrics,
            save_path=Metric.get_save_path(
                model_args.eval_data,
                training_args.output_dir
            ) if model_args.eval_data is not None else None
        )
    )
    
    if train_dataset is not None:
        logger.info(f"Start Training...")
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            try:
                trainer.train(resume_from_checkpoint=checkpoint)
            except Exception as e:
                logger.error(f"[Rank {training_args.local_rank}] CRITICAL ERROR during training: {type(e).__name__}: {str(e)}")
                logger.error(f"[Rank {training_args.local_rank}] Traceback:\n{traceback.format_exc()}")
                raise e
    elif eval_dataset is not None:
        logger.info("Starting evaluation...")
        trainer.evaluate()

if __name__ == "__main__":
    main()
