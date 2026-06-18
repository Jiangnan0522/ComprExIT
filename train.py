
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation


# Arguments
## Args related to number of samples
    - num_samples: number of total samples limited from the raw dataset (before processing, after loading). *Only for debugging purposes.*
    - max_tokens: max number of tokens during processing (during processing)
    - max_train_samples: number of training samples limited from the processed dataset (after processing)
    - max_eval_samples: number of evaluation samples limited from the processed dataset (after processing)
    
    (Normally, you do not need to set max_train_samples and max_eval_samples if you set max_tokens, 
        since max_tokens affect pretraining, max_train_samples and max_eval_samples affect SFT.)
"""


import logging
import math
import traceback
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
from loguru import logger
from datetime import datetime

import datasets
import torch
from datasets import IterableDataset, IterableDatasetDict, load_dataset

from peft import get_peft_model, PromptTuningConfig, TaskType

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from src.data_processing.data_loading import data_loading_factory, sft_data_loading_factory
from src.data_processing.ntp_preprocessing import GistDataProcessor, get_data_collator_factory
from src.data_processing.sft_preprocessing import SFTDataProcessor, DataCollatorForSFT, DataCollatorForSFTBaseModel
from src.model.model import get_model_factory, get_model_factory_from_config
from src.training import CompressInTrainer, LoguruCallback, DeviceUsageCallback
from src.device_utils import get_device_module


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
PAD_TOKEN = "<pad>"

device_module, device_type = get_device_module()



@dataclass
class TrainingArgs(TrainingArguments):
    mode:str = field(default="ntp", metadata={"help": "The mode of the training. Options: 'sft', 'ntp'."})
    model_structure:str = field(default="hier", metadata={"help": "The structure of the model to use."})
    ddp_backend: str = field(default="ccl" if device_type == "xpu" else "nccl", metadata={"help": "The backend to use for distributed training."})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Whether to remove unused columns. Data collator will handle this."}) 
    bf16: bool = field(default=False, metadata={"help": "Whether to use bf16 for mixed-precision training (autocast)."})
    report_to: str = field(default="wandb", metadata={"help": "The report to use for training."})
    save_strategy: str = field(default="steps", metadata={"help": "The strategy to use for saving the model."})
    save_steps: int = field(default=500, metadata={"help": "The number of steps to save the model."})
    save_total_limit: int = field(default=1, metadata={"help": "Maximum number of checkpoints to keep. Older checkpoints are deleted."})
    run_name: str = field(default=None, metadata={"help": "The name of the run."})
    logging_steps: int = field(default=100, metadata={"help": "The number of steps to log the training progress."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "The maximum gradient norm to clip."})

    # --- New arguments ---
    shuffle_train_set: bool = field(default=True, metadata={"help": "Whether to shuffle the training set."})
    ## baseline finetuning (promp tuning the raw model)
    sft_base_model: bool = field(default=False, metadata={"help": "Whether to use the raw model for finetuning."})
    num_virtual_tokens: int = field(default=32, metadata={"help": "The number of virtual tokens to use for the raw model."})
    # map reduce training
    map_reduce_training: bool = field(default=False, metadata={"help": "Whether to use map-reduce training."})
    # direct SFT (skip NTP pretraining)
    direct_sft: bool = field(default=False, metadata={"help": "Initialize model from scratch (base LLM) for direct SFT without NTP pretraining."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # --- Model internal arguments ---
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."},
    )
    top_k_layers: int = field(default=10, metadata={"help": "The number of layers to compress."})
    num_hidden_layers: int = field(default=2, metadata={"help": "The number of hidden layers in the projection MLP."})
    attn_implementation: str = field(default="sdpa" if device_type == "xpu" else "flash_attention_2", metadata={"help": "The implementation of the attention mechanism."})

    # freezing mode
    training_freezing_mode: str = field(
        default="compress+llm",
        metadata={
            "help": "Training mode that determines which components to freeze. Options: 'compress+llm' (or 'both'), 'compress', 'llm', 'projector', 'none'",
            "choices": ["compress+llm", "both", "compress", "llm", "projector", "none"]
        }
    )
    
    # --- Pooling arguments ---
    pooling_method: str = field(default="sliding", metadata={"help": "The pooling method to use."})
    compression_ratio: int = field(default=4, metadata={"help": "The compression ratio for pooling."})
    add_global_avg: bool = field(default=False, metadata={"help": "Whether to add global average pooling."})
    projector_gain: float = field(default=0.4, metadata={"help": "The gain for the projector initialization."})
    layerwise_pooling_layers: Optional[List[int]] = field(default=None, metadata={"help": "The layers to pool."})
    layerwise_pooling_gate_hidden: int = field(default=256, metadata={"help": "The hidden size for the gating MLP."})
    layerwise_pooling_temperature: float = field(default=0.7, metadata={"help": "The temperature for the softmax function."})
    chunk_attn_hidden_size: int = field(default=256, metadata={"help": "The hidden size for the chunk attention."})
    chunk_attn_num_heads: int = field(default=4, metadata={"help": "The number of heads for chunk attention."})
    reduced_hidden_size: int = field(default=64, metadata={"help": "The hidden size for the reduced hidden states."})
    # --- For OptimalTransportPooling ("ot") ---
    ot_window_size: int = field(default=128, metadata={"help": "The window size for the OT pooling."})
    ot_n_iter: int = field(default=30, metadata={"help": "The number of iterations for the Sinkhorn algorithm."})
    ot_metric_dim: int = field(default=256, metadata={"help": "The hidden size for the metric embeddings."})
    ## OT Ablations
    ab_ot_shuffle_anchors: bool = field(default=False, metadata={"help": "Whether to shuffle the anchor order for the OT pooling."})

    # ICAE arguments
    gist_token:str=field(default='<gist>', metadata={"help": "The token to use for the gist tokens."})

    # For Map-Reduce 
    map_reduce_seg_len: int = field(default=512, metadata={"help": "The segment length for the map-reduce compression."})
    
    # for lora
    lora_compressor:bool=field(default=True, metadata={"help": "Whether to use LoRA for the compressor."})
    lora_r:int=field(default=128, metadata={"help": "The rank of the LoRA matrix."})
    lora_alpha:int=field(default=32, metadata={"help": "The alpha of the LoRA matrix."})
    lora_dropout:float=field(default=0.05, metadata={"help": "The dropout of the LoRA matrix."})
    lora_bias:str=field(default="none", metadata={"help": "The bias of the LoRA matrix."})
    lora_task_type:str=field(default="CAUSAL_LM", metadata={"help": "The task type of the LoRA matrix."})
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with LoRA."})

    # --- Model external arguments ---
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    num_memory_tokens: Optional[int] = field(default=64, metadata={"help": "The number of memory tokens to use for ICAE model."})


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder of the dataset to use."}
    )
    sft_dataset_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The names of the SFT datasets to use."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of samples to this "
                "value if set."
            )
        },
    )
    skip_train_files: int = field(
        default=0,
        metadata={
            "help": (
                "Number of train data files to skip from the beginning. "
                "Useful for continued pretraining on later portions of the data "
                "(e.g., for SlimPajama 6B with 48 files, set to 8 to skip ~1B tokens). "
                "Files are skipped before any processing, so this is O(1) with no overhead."
            )
        },
    )
    max_tokens: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of tokens to this "
                "value if set."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    context_length: Optional[int] = field(default=256, metadata={"help": "The context length of the input text."})
    generation_length: Optional[int] = field(default=256, metadata={"help": "The generation length of the input text."})
    ntp_ratio: float = field(
        default=1.0,
        metadata={"help": "Proportion of samples allocated to next-token prediction (0-1)."},
    )
    prompt: Optional[str] = field(
        default="Repeat the previous content:",
        metadata={"help": "Optional prompt prepended to reconstruction samples. If None, no prompt will be prepended."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    num_gist_tokens: Optional[int] = field(default=64, metadata={"help": "The number of gist tokens to use for ICAE-flex model."})


def split_streaming_dataset(
    full_streaming_dataset,
    validation_percentage: int = 5,
) -> IterableDatasetDict:
    """
    Splits a streaming dataset into
    training and validation IterableDatasets, and supports methods like .map(), .filter(),
    .take() and properties like .features on the resulting streams.

    Args:
        full_streaming_dataset (Dataset): The name of the dataset to load (e.g., "HuggingFaceFW/fineweb").
        validation_percentage (int): The proportion of the dataset to be used for validation split.

    Returns:
        IterableDatasetDict: An IterableDatasetDict containing two IterableDataset objects: (train_stream, validation_stream).
    """
    if not (0 < validation_percentage < 100):
        raise ValueError(
            f"validation_percentage must be between 0 and 100 (exclusive). Passed: {validation_percentage}"
        )

    def split_generator(is_train: bool):
        for i, example in enumerate(full_streaming_dataset):
            if is_train:
                if i % 100 > validation_percentage:
                    yield example
            else:
                if i % 100 < validation_percentage:
                    yield example

    features = full_streaming_dataset.features
    train_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": True}, features=features)
    validation_stream = IterableDataset.from_generator(
        split_generator, gen_kwargs={"is_train": False}, features=features
    )

    return IterableDatasetDict({"train": train_stream, "validation": validation_stream})



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Add Liger Kernel support
    if device_type == 'cuda':
        if 'qwen' in model_args.model_name_or_path.lower():
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_qwen
                apply_liger_kernel_to_qwen()
                logger.info("Liger Kernels applied successfully for Qwen.")
            except Exception as e:
                logger.error(f"Failed to apply Liger Kernels for Qwen: {e}")
        elif 'llama' in model_args.model_name_or_path.lower():
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama()
                logger.info("Liger Kernels applied successfully for Llama.")
            except Exception as e:
                logger.error(f"Failed to apply Liger Kernels for Llama: {e}")
        else:
            logger.warning(f"No Liger Kernels found for {model_args.model_name_or_path}.")

    # --- DDP GPU mapping safety ---
    # On some clusters, NCCL can hang if the process hasn't explicitly selected its GPU
    # before hitting any distributed collectives/barriers. Force device = LOCAL_RANK.
    if device_type == "cuda" and torch.cuda.is_available():
        try:
            env_local_rank = os.environ.get("LOCAL_RANK")
            if env_local_rank is not None and str(env_local_rank) != "":
                torch.cuda.set_device(int(env_local_rank))
                logger.info(f"[DDP] Set CUDA device to LOCAL_RANK={env_local_rank}")
        except Exception as e:
            logger.warning(f"[DDP] Failed to set CUDA device from LOCAL_RANK: {e}")

    # SFT preprocessing is map-style only (DatasetDict), not streaming.
    if training_args.mode == "sft" and data_args.streaming:
        logger.warning("--streaming cannot be used with --mode sft. setting streaming to False.")
        data_args.streaming = False

    if data_args.streaming:
        if data_args.max_tokens is None:
            raise ValueError("max_tokens must be set when streaming is True in order to calculate the number of steps.")
        # max_steps will be calculated later after processing the dataset
    
    # Avoid race conditions where multiple ranks try to create the same output dir at once.
    os.makedirs(training_args.output_dir, exist_ok=True)

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

        
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if device_module:
        logger.info(f"{device_type.upper()} Available. Device count: {device_module.device_count()}")
    else:
        logger.warning("No XPU or CUDA device detected.")

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    
    # Set seed before initializing model.
    set_seed(training_args.seed)


    # Get the datasets from files
    if training_args.mode == "ntp":
        dataset_name = os.path.basename(data_args.dataset_folder)
        raw_datasets, columns_to_remove = data_loading_factory(dataset_name, data_folder=data_args.dataset_folder, streaming=data_args.streaming, skip_train_files=data_args.skip_train_files)
    elif training_args.mode == "sft":
        if data_args.sft_dataset_names is None:
            raise ValueError("sft_dataset_names must be set when --mode is sft.")
        dataset_name = data_args.sft_dataset_names
        raw_datasets, columns_to_remove = sft_data_loading_factory(dataset_names=data_args.sft_dataset_names, datasets_dir=data_args.dataset_folder)
    else:
        raise ValueError(f"Invalid --mode: {training_args.mode}. Expected 'ntp' or 'sft'.")

    if (data_args.num_samples is not None) and (data_args.num_samples > 0):
        if data_args.streaming:
            raw_datasets = IterableDatasetDict({
                k: v.take(data_args.num_samples) for k, v in raw_datasets.items()
            })
        else:
            raw_datasets = datasets.DatasetDict({
                k: v.select(range(min(len(v), data_args.num_samples))) for k, v in raw_datasets.items()
            })
        logger.info(f"Num_samples is set to {data_args.num_samples}. Therefore, selected {data_args.num_samples} samples for each split.")


    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # --- Detect whether model_name_or_path is a CompressIn checkpoint or a base LLM ---
    # If it's a CompressIn checkpoint (continued NTP), we load all weights directly.
    # If it's a base LLM (fresh NTP), we build from scratch via from_pretrained_submodules.
    is_compressin_checkpoint = False
    if training_args.mode == 'ntp' and model_args.model_name_or_path:
        _config_path = os.path.join(model_args.model_name_or_path, "config.json")
        if os.path.exists(_config_path):
            try:
                get_model_factory_from_config(_config_path)
                is_compressin_checkpoint = True
                logger.info(f"Detected CompressIn checkpoint at {model_args.model_name_or_path}. Will load full model weights for continued NTP.")
            except ValueError:
                pass  # Not a CompressIn config — treat as base LLM

    # --- Shared fresh-init path: build backbone_config + model_config from base LLM ---
    # Used by both NTP (fresh) and direct-SFT (initializing a fresh CompressIn model from a base LLM).
    fresh_init = (training_args.mode == 'ntp' and not is_compressin_checkpoint) or training_args.direct_sft
    if fresh_init:
        backbone_config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.token,
            "trust_remote_code": model_args.trust_remote_code,
        }
        if model_args.config_name:
            backbone_config = AutoConfig.from_pretrained(model_args.config_name, **backbone_config_kwargs)
        elif model_args.model_name_or_path:
            backbone_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **backbone_config_kwargs)
        else:
            backbone_config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                backbone_config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {backbone_config}")

        structure = get_model_factory(training_args.model_structure)
        model_config = structure['config'](
            lm_name_or_path=model_args.model_name_or_path,
            dtype=model_args.dtype,
            attn_implementation=model_args.attn_implementation,
            # for compressor
            top_k_layers=model_args.top_k_layers,
            pooling_method=model_args.pooling_method,
            context_length=data_args.context_length,
            training_freezing_mode=model_args.training_freezing_mode,
            ## for sliding window pooling
            compression_ratio=model_args.compression_ratio,
            ## For TokenLevelLayerwisePooling
            layerwise_pooling_layers=model_args.layerwise_pooling_layers,
            layerwise_pooling_gate_hidden=model_args.layerwise_pooling_gate_hidden,
            layerwise_pooling_temperature=model_args.layerwise_pooling_temperature,
            chunk_attn_num_heads=model_args.chunk_attn_num_heads,
            chunk_attn_hidden_size=model_args.chunk_attn_hidden_size,
            reduced_hidden_size=model_args.reduced_hidden_size,
            # --- For OptimalTransportPooling ("ot") ---
            ot_window_size=model_args.ot_window_size,
            ot_n_iter=model_args.ot_n_iter,
            ot_metric_dim=model_args.ot_metric_dim,
            ## OT Ablations
            ab_ot_shuffle_anchors=model_args.ab_ot_shuffle_anchors,
            # for projector
            num_hidden_layers=model_args.num_hidden_layers,
            add_global_avg=model_args.add_global_avg,
            projector_gain=model_args.projector_gain,
            # for ICAE-flex
            gist_token=model_args.gist_token,
            # for ICAE
            num_memory_tokens=model_args.num_memory_tokens,
            # for lora
            lora_target_modules=model_args.lora_target_modules,
            lora_compressor=model_args.lora_compressor,
            lora_r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            lora_bias=model_args.lora_bias,
            lora_task_type=model_args.lora_task_type,
            # for map-reduce
            map_reduce_seg_len=model_args.map_reduce_seg_len,
            # backbone info
            backbone_config_dict=backbone_config.to_dict() # pass the config of the backbone for initiliazation
        )
        model_config.attn_implementation = model_args.attn_implementation # ensure this is passed to the config

    if is_compressin_checkpoint:
        # --- Continued NTP: load full CompressIn model from checkpoint ---
        logger.info(f"Loading CompressIn checkpoint for continued NTP from {model_args.model_name_or_path}.")
        model_factory = get_model_factory_from_config(os.path.join(model_args.model_name_or_path, "config.json"))
        model_class, config_class = model_factory["class"], model_factory["config"]

        model_config = config_class.from_pretrained(model_args.model_name_or_path)
        # Override config fields that may change between training stages
        model_config.context_length = data_args.context_length
        model_config.training_freezing_mode = model_args.training_freezing_mode
        model_config.attn_implementation = model_args.attn_implementation
        model_config.dtype = model_args.dtype

        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(model_args.dtype, torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

        logger.info(f"[Rank {training_args.local_rank}] About to load model with config: {model_config}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            low_cpu_mem_usage=True,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
        )
        logger.info(f"[Rank {training_args.local_rank}] CompressIn checkpoint loaded successfully for continued NTP.")

    elif training_args.mode == 'ntp' or training_args.direct_sft:
        logger.info(f"Attn implementation: {model_args.attn_implementation}")
        logger.info(f"Loading model from {model_args.model_name_or_path}.")
        logger.info(f"[Rank {training_args.local_rank}] About to load model with config: {model_config}")

        try:
            # Log device memory before loading
            if device_module:
                for i in range(device_module.device_count()):
                    mem_allocated = device_module.memory_allocated(i) / 1024**3
                    mem_reserved = device_module.memory_reserved(i) / 1024**3
                    logger.info(f"[Rank {training_args.local_rank}] {device_type.upper()} {i} memory before model loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

            if training_args.sft_base_model and training_args.mode == 'ntp':
                raise ValueError("SFT base model is only supported for SFT.")

            model = structure['class'].from_pretrained_submodules(config=model_config, tokenizer=tokenizer)
            logger.info(f"[Rank {training_args.local_rank}] Model loaded successfully.")

            # Log device memory after loading
            if device_module:
                for i in range(device_module.device_count()):
                    mem_allocated = device_module.memory_allocated(i) / 1024**3
                    mem_reserved = device_module.memory_reserved(i) / 1024**3
                    logger.info(f"[Rank {training_args.local_rank}] {device_type.upper()} {i} memory after model loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        except Exception as e:
            logger.error(f"[Rank {training_args.local_rank}] CRITICAL ERROR during model loading: {type(e).__name__}: {str(e)}")
            logger.error(f"[Rank {training_args.local_rank}] Traceback:\n{traceback.format_exc()}")
            raise

        # For direct_sft: handle pad token (same as SFT checkpoint path)
        if training_args.direct_sft:
            if (tokenizer.pad_token_id) is None or (tokenizer.pad_token_id == tokenizer.eos_token_id):
                tokenizer.add_special_tokens({"additional_special_tokens": [PAD_TOKEN]})
                tokenizer.pad_token = PAD_TOKEN
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
            model.tokenizer = tokenizer # update the tokenizer in the model.

    elif training_args.mode == 'sft':
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(model_args.dtype, torch.float32)
        if not training_args.sft_base_model:
            model_factory = get_model_factory_from_config(os.path.join(model_args.model_name_or_path, "config.json"))
            model_class, config_class = model_factory["class"], model_factory["config"]

            model_config = config_class.from_pretrained(model_args.model_name_or_path)
            model_config.attn_implementation = model_args.attn_implementation

            logger.info(f"Loading model from {model_args.model_name_or_path}.")
            logger.info(f"[Rank {training_args.local_rank}] About to load model with config: {model_config}")

            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
                low_cpu_mem_usage=True,
                tokenizer=tokenizer,
                torch_dtype=torch_dtype,
            )
            logger.info(f"[Rank {training_args.local_rank}] Model loaded successfully.")
            # add pad token if not exists for sft.
            tokenizer = model.tokenizer # use the tokenizer from the model in case any changes are made to the tokenizer.
        else:
            # prompt tuning
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
                low_cpu_mem_usage=True,
                device_map="cpu"  # Load to CPU first, let Trainer move to XPU
            )
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=training_args.num_virtual_tokens,
                tokenizer_name_or_path=model_args.model_name_or_path,
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text="Answer the question directly with a short span only, no explanation. Answer:\n"
            )
            model = get_peft_model(model, peft_config)
            logger.info(f"[Rank {training_args.local_rank}] Prompt tuning model(peft) loaded successfully.")

        if (tokenizer.pad_token_id) is None or (tokenizer.pad_token_id == tokenizer.eos_token_id):
            tokenizer.add_special_tokens({"additional_special_tokens": [PAD_TOKEN]})
            tokenizer.pad_token = PAD_TOKEN
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        model.tokenizer = tokenizer # update the tokenizer in the model.
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Set gradient stopping based on training mode
    logger.info(f"Setting freezing mode: {model_args.training_freezing_mode}")

    if not training_args.sft_base_model:
        model.stop_gradient(mode=model_args.training_freezing_mode)


    # ✅ Preprocessing the datasets.
    # Manual synchronization to avoid issues with main_process_first on Intel XPU
    logger.info(f"[Rank {training_args.local_rank}] About to process dataset")    
    dist_inited = bool(getattr(torch, "distributed", None)) and torch.distributed.is_available() and torch.distributed.is_initialized()
    logger.info(f"[Rank {training_args.local_rank}] Entering main_process_first (torch.distributed initialized={dist_inited})")
    with training_args.main_process_first():
        logger.info(f"[Rank {training_args.local_rank}] Main process: processing dataset")
        if training_args.mode == "ntp":
            processor = GistDataProcessor(
                tokenizer=tokenizer,
                context_length=data_args.context_length,
                generation_length=data_args.generation_length,
                seed=training_args.seed,
                ntp_ratio=data_args.ntp_ratio,
            )
            lm_datasets = processor.process_dataset(
                raw_datasets=raw_datasets,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                overwrite_cache=data_args.overwrite_cache,
                streaming=data_args.streaming,
                columns_to_remove=columns_to_remove,
                shuffle_train_set=training_args.shuffle_train_set,
                max_tokens=data_args.max_tokens,
                max_train_samples=data_args.max_train_samples,
                max_eval_samples=data_args.max_eval_samples,
            )
        else:
            processor = SFTDataProcessor(
                tokenizer=tokenizer,
                max_context_length=data_args.context_length,
                max_generation_length=data_args.generation_length,
                seed=training_args.seed,
                overwrite_cache=data_args.overwrite_cache,
            )
            # SFT processor returns a standard (map-style) DatasetDict with fixed-length fields.
            lm_datasets = processor.process_dataset(
                dataset=raw_datasets, 
                shuffle_train_set=training_args.shuffle_train_set,
                overwrite_cache=data_args.overwrite_cache,
            )
        logger.info(f"[Rank {training_args.local_rank}] Main process: dataset processing completed")
    logger.info(f"[Rank {training_args.local_rank}] Exited main_process_first; continuing setup")
    

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]


    # Calculate the number of steps (streaming-only path; SFT is non-streaming)
    if (training_args.max_steps is None or training_args.max_steps < 0) and data_args.streaming:
        actual_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        max_steps_per_epoch = math.ceil(processor.max_samples / actual_batch_size) 
        max_steps = math.ceil(max_steps_per_epoch * training_args.num_train_epochs )   
        training_args.max_steps = max_steps
        logger.info(f"[Max steps] Calculated max steps in each epoch: {max_steps_per_epoch}")
        logger.info(f"[Max steps] Calculated max steps in total: {max_steps}")


    # Initialize our Trainer
    # Set label_names to tell Trainer which keys contain labels
    training_args.label_names = ["labels"]
    if training_args.mode == "ntp":
        collator = get_data_collator_factory(training_args.model_structure)(
            tokenizer,
            context_length=data_args.context_length,
            num_gist_tokens=data_args.num_gist_tokens,
            prompt=data_args.prompt,
        )
    else:
        if not training_args.sft_base_model:
            collator = DataCollatorForSFT()
        else:
            collator = DataCollatorForSFTBaseModel()

    # Map reduce training
    if training_args.map_reduce_training:
        model.enable_map_reduce()
        model_config.trained_with_map_reduce = True
    else:
        if hasattr(model, 'disable_map_reduce'):
            model.disable_map_reduce()

    trainer = CompressInTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=collator,
        # Extra context for SFT evaluation (used by CompressInTrainer.evaluate)
        sft_dataset_names=data_args.sft_dataset_names if training_args.mode == "sft" else None,
        sft_datasets_dir=data_args.dataset_folder if training_args.mode == "sft" else None,
        callbacks=[LoguruCallback(), DeviceUsageCallback(training_args.local_rank, training_args.logging_steps)],
    )

    # Training
    logger.info(f"Start Training...")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            # trainer.save_model()  # Saves the tokenizer too for easy upload
        except Exception as e:
            logger.error(f"[Rank {training_args.local_rank}] CRITICAL ERROR during training: {type(e).__name__}: {str(e)}")
            logger.error(f"[Rank {training_args.local_rank}] Traceback:\n{traceback.format_exc()}")
            raise e

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        if training_args.mode == "ntp":
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
        elif training_args.mode == "sft":
            metrics = metrics
        else:
            raise ValueError(f"Invalid mode: {training_args.mode}")

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info(f"Evaluation Results: {metrics}")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_folder is not None:
        kwargs["dataset_tags"] = dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)



if __name__ == "__main__":
    # Intel XPU DDP setup: map MPI environment variables to PyTorch Distributed variables
    if device_type == "xpu":
        try:
            import intel_extension_for_pytorch
            import oneccl_bindings_for_pytorch
        except ImportError:
            raise ImportError("Intel XPU DDP setup failed. Please install intel_extension_for_pytorch and oneccl_bindings_for_pytorch.")

        def get_int_from_env(env_keys, default):
            for key in env_keys:
                if key in os.environ:
                    return int(os.environ[key])
            return int(default)

        # If running with MPI (e.g. mpirun), set the necessary environment variables for DDP
        if "PMI_SIZE" in os.environ or "MPI_LOCALRANKID" in os.environ or "OMPI_COMM_WORLD_SIZE" in os.environ:
            local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID", "PMI_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"], "0")
            world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE"], "1")
            rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK"], "0")
            port = get_int_from_env(["MASTER_PORT"], 29500)

            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(rank)
            os.environ["MASTER_PORT"] = str(port)
            
            # Default MASTER_ADDR if not set
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                
            # Set the device for this process
            if device_module:
                dev_count = device_module.device_count()
                logger.info(f"{device_type.upper()} device count: {dev_count}")
                if dev_count > 0:
                    # If only one device is visible (e.g. via isolation), use device 0
                    # Otherwise use local_rank if multiple devices are visible
                    device_id = local_rank % dev_count
                    device_str = f"{device_type}:{device_id}"
                    
                    if device_type == "xpu":
                        torch.xpu.set_device(device_str)
                    elif device_type == "cuda":
                        torch.cuda.set_device(device_id) # cuda set_device takes int or device object

                    logger.info(f"Set {device_type.upper()} device to {device_str} for local_rank {local_rank}")
            
            logger.info(f"DDP Setup: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={port}")
    
    # Run
    main()