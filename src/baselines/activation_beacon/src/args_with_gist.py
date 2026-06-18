"""
Extended arguments for training with GistDataProcessor.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from .args import ModelArgs, TrainingArgs as BaseTrainingArgs


@dataclass
class GistTrainingArgs(BaseTrainingArgs):
    """
    Extended training arguments that include GistDataProcessor-specific parameters.
    """

    # Training mode
    mode: str = field(
        default="ntp",
        metadata={
            "help": "Training mode. 'ntp' = next-token prediction on plain text blocks (current default). "
                    "'sft' = supervised fine-tuning on QA-style datasets (context/question/gold) or instruction data.",
            "choices": ["ntp", "sft"],
        },
    )
    
    # GistDataProcessor specific arguments
    ntp_ratio: float = field(
        default=1.0,
        metadata={'help': 'Ratio of next-token prediction samples vs reconstruction samples. Default 1.0 means all NTP.'}
    )
    
    context_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Length of text to compress (Input to Encoder). If None, will be set to max_length // 2.'}
    )
    
    generation_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Length of text to predict (Target for Decoder). If None, will be set to max_length - context_length.'}
    )
    
    streaming: bool = field(
        default=True,
        metadata={'help': 'Whether to use streaming mode for datasets.'}
    )
    
    max_tokens: Optional[List[int]] = field(
        default=None,
        metadata={'help': 'Maximum number of tokens (in millions) for [train, validation] splits. E.g., [3000, 1000] for 3B train and 1B validation tokens.'}
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of training samples. Overrides max_tokens if set.'}
    )
    
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of workers for preprocessing (only used in non-streaming mode).'}
    )
    
    overwrite_cache: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite the preprocessing cache.'}
    )
    
    # Data loading arguments (compatible with your train.py)
    dataset_folder: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the folder containing the downloaded HuggingFace dataset. If provided, will use data_loading_factory instead of train_data/eval_data.'}
    )

    # SFT dataset selection (when mode='sft' and using dataset_folder)
    sft_dataset_names: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Names of SFT datasets to load from --dataset_folder when --mode sft. "
                    "Matches the dataset classes in src/data_processing/sft_datasets.py (e.g., ['squad'])."
        },
    )
    
    num_samples: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of samples to limit from raw dataset (before processing). Only for debugging purposes.'}
    )


__all__ = ['ModelArgs', 'GistTrainingArgs']
