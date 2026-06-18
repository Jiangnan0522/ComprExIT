from abc import ABC, abstractmethod
import hashlib
import math
import random

import torch
from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset, DatasetDict, IterableDatasetDict, Dataset
from itertools import chain
from typing import Dict, List, Any, Optional, Union
from loguru import logger


def get_data_collator_factory(model_structure: str):
    if model_structure == 'hier':
        return DataCollatorForHierarchicalCompressor
    elif model_structure == 'icae-flex':
        return DataCollatorForICAEFlex
    elif model_structure == 'icae':
        return DataCollatorForICAE
    elif model_structure == '500x':
        return DataCollatorFor500x
    elif model_structure == 'sac':
        return DataCollatorFor500x
    else:
        raise ValueError(f"Invalid model structure: {model_structure}")

class DataCollatorBase(ABC):
    name = 'base'
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __call__(self, features) -> Dict[str, torch.Tensor]:
        pass


class GistDataProcessor:
    def __init__(
        self, 
        tokenizer, 
        context_length: int = 256, 
        generation_length: int = 256,
        seed: int = 42,
        ntp_ratio: float = 1.0,
    ):
        """
        Args:
            tokenizer: The HF tokenizer.
            context_length: Length of text to compress (Input to Encoder).
            generation_length: Length of text to predict (Target for Decoder).

        NOTE: Currently context length and generation length are not reflected in the code, 
            which are only used to calculate the total sequence length.
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.generation_length = generation_length
        self.total_seq_length = context_length + generation_length
        self._max_samples = None
        self.seed = seed
        # Maintain the actual ntp ratio because one sequence yields two reconstruction samples
        self.ntp_ratio = 2 * ntp_ratio / (1 + ntp_ratio) 
        self._num_reconstruction_segments = max(1, math.ceil(self.total_seq_length / self.context_length))
        self._task_cycle_length = 1000

    @property
    def max_samples(self):
        return self._max_samples

    def _tokenize_function(self, examples):
        # Tokenize raw text without padding; we will handle concatenation later
        return self.tokenizer(examples["text"], truncation=False, padding=False)

    def _group_texts(self, examples):
        """
        - Concatenates dataset samples in the batch
        - Drops the small remainder at the end to keep shapes constant
        - chunks them into blocks of `total_seq_length`.
        
        Each sample will have the same length of input ids and attention mask. 
        This ensures we don't waste tokens on padding.

        We assume the ids are stored in the 'input_ids' key of the examples.

        Args:
            examples: tokenized examples in a batch 
        """

        input_ids_key = "input_ids"

        if input_ids_key not in examples:
            raise ValueError("input_ids not found in examples, which can mean the examples are not tokenized.")

        # Concatenate all elements from the batch
        # Only process keys where the value is a list (sequence) to avoid chaining scalars
        concatenated = {}
        keys_to_skip_processing = []
        for k in examples.keys():
            if len(examples[k]) > 0 and isinstance(examples[k][0], list):
                concatenated[k] = list(chain(*examples[k]))
            else:
                # only concate sequences in the examples
                concatenated[k] = examples[k]
                keys_to_skip_processing.append(k)
        
        # Calculate the total length of the concatenated elements using the input ids.
        total_length = len(concatenated[input_ids_key])
        # We drop the small remainder at the end to keep shapes constant
        if total_length >= self.total_seq_length:
            total_length = (total_length // self.total_seq_length) * self.total_seq_length
            
        # Split by total_seq_length (for example, split input_ids and attention masks)
        result = {}
        for k, t in concatenated.items():
            if k not in keys_to_skip_processing:
                result[k] = [t[i : i + self.total_seq_length] for i in range(0, total_length, self.total_seq_length)]
            else:
                result[k] = t
        return result

    def _split_seed(self, split: str) -> int:
        """Derive a deterministic per-split seed so ntp assignments stay reproducible."""
        digest = hashlib.sha1(f"{split}-{self.seed}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _build_ntp_flags(self, total: int, split: str) -> List[bool]:
        """Randomly mark samples as ntp according to ntp_ratio while keeping counts reproducible per split."""
        ratio = self.ntp_ratio
        if total == 0:
            return []
        if ratio <= 0.0:
            return [False] * total
        if ratio >= 1.0:
            return [True] * total
        indices = list(range(total))
        rng = random.Random(self._split_seed(split))
        rng.shuffle(indices)
        cutoff = int(round(total * ratio))
        cutoff = max(0, min(total, cutoff))
        ntp_set = set(indices[:cutoff])
        return [idx in ntp_set for idx in range(total)]

    def _build_recon_segments(self, flags: List[bool]) -> List[int]:
        """Assign reconstruction samples to segments so each chunk can yield multiple reconstruction views."""
        if not flags:
            return []
        num_segments = self._num_reconstruction_segments
        recon_idx = 0
        segments = []
        for is_ntp in flags:
            if is_ntp:
                segments.append(-1)
            else:
                segments.append(recon_idx % num_segments)
                recon_idx += 1
        return segments

    def _assign_task_metadata_map_style(self, dataset: Dataset, split: str) -> Dataset:
        """Attach for_ntp/reconstruction_segment columns for map-style datasets."""
        total = len(dataset)
        flags = self._build_ntp_flags(total, split)
        segments = self._build_recon_segments(flags)
        num_segments = self._num_reconstruction_segments

        expanded_indices: List[int] = []
        expanded_for_ntp: List[bool] = []
        expanded_segments: List[int] = []

        for idx, is_ntp in enumerate(flags):
            if is_ntp:
                expanded_indices.append(idx)
                expanded_for_ntp.append(True)
                expanded_segments.append(-1)
            else:
                start_segment = segments[idx] if segments else 0
                for offset in range(num_segments):
                    expanded_indices.append(idx)
                    expanded_for_ntp.append(False)
                    expanded_segments.append((start_segment + offset) % num_segments)

        # Expand the dataset because of the reconstruction samples.
        if expanded_indices:
            dataset = dataset.select(expanded_indices)
        else:
            dataset = dataset.select([])

        for column in ["for_ntp", "reconstruction_segment"]:
            if column in dataset.column_names:
                dataset = dataset.remove_columns([column])

        dataset = dataset.add_column("for_ntp", expanded_for_ntp)
        dataset = dataset.add_column("reconstruction_segment", expanded_segments)
        return dataset

    def _assign_task_metadata_iterable(self, dataset: IterableDataset, split: str) -> IterableDataset:
        """Attach task metadata for streaming datasets using a simple cyclic pattern."""
        ratio = self.ntp_ratio
        num_segments = self._num_reconstruction_segments
        cycle = max(1, self._task_cycle_length)
        ntp_per_cycle = int(round(cycle * ratio))
        ntp_per_cycle = max(0, min(cycle, ntp_per_cycle))

        def generator():
            counters = {"idx": 0, "recon": 0}
            for example in dataset:
                example = dict(example)
                if ratio <= 0.0:
                    is_ntp = False
                elif ratio >= 1.0:
                    is_ntp = True
                else:
                    position = counters["idx"] % cycle
                    is_ntp = position < ntp_per_cycle
                counters["idx"] += 1

                if is_ntp:
                    example["for_ntp"] = True
                    example["reconstruction_segment"] = -1
                    yield example
                else:
                    start_segment = counters["recon"] % num_segments
                    counters["recon"] += 1
                    for offset in range(num_segments):
                        seg_example = dict(example)
                        seg_example["for_ntp"] = False
                        seg_example["reconstruction_segment"] = (start_segment + offset) % num_segments
                        yield seg_example

        return IterableDataset.from_generator(generator)

    def _assign_task_metadata(self, datasets_dict: Dict[str, Union[Dataset, IterableDataset]]) -> Dict[str, Union[Dataset, IterableDataset]]:
        """Wrapper that dispatches to map/iterable helpers.

        The preprocessing pipeline receives split->dataset mappings that can be either
        Hugging Face map-style datasets or streaming iterable datasets. This helper
        inspects each split and forwards it to the appropriate metadata injection
        routine so every split ends up with consistent `for_ntp` and`reconstruction_segment` fields, 
        regardless of the underlying dataset type.

        Reconstruction-designated samples are duplicated once per reconstruction
        segment so every chunk contributes supervision for each slice rather than a
        single segment.
        
        Fields
            - `for_ntp`: indicates whether an example participates in next-token prediction (True) or is reserved for reconstruction (False). 
            - `reconstruction_segment`: stores the segment index used to bucket reconstruction examples (and is -1
            for pure NTP examples), so downstream losses know how to regroup them.

            NOTE: The length of the segment is default to be the same as the **context length**. We only record the segment index here.
            You have to use the context length to get the actual segment.
        """
        updated = {}
        for split, dataset in datasets_dict.items():
            if isinstance(dataset, Dataset):
                updated[split] = self._assign_task_metadata_map_style(dataset, split)
            else:
                updated[split] = self._assign_task_metadata_iterable(dataset, split)
        return updated


    def process_dataset(
        self,
        raw_datasets: Union[DatasetDict, IterableDatasetDict],
        text_column_name: str = "text",
        columns_to_remove: Optional[List[str]] = None,
        streaming: bool = True,
        preprocessing_num_workers: Optional[int] = None,
        overwrite_cache: bool = False,
        shuffle_train_set: bool = True,
        # Args related to number of samples
        max_tokens: Optional[list[int]] = None,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
    ) -> Dict[str, IterableDataset]:
        """
        Processes a dataset with splits (tokenization and grouping).

        Args:
            raw_datasets: Loaded dataset with splits (e.g., DatasetDict or IterableDatasetDict).
            text_column_name: Name of the text column to process.
            columns_to_remove: List of column names to remove (e.g., ["meta", "redpajama_set_name"]).
                               If None, will only remove the text column after tokenization.
            streaming: Whether the dataset is in streaming mode.
            preprocessing_num_workers: Number of workers for preprocessing (only used for non-streaming).
            overwrite_cache: Whether to overwrite the cache.
            shuffle_train_set: Whether to shuffle the training set.
            max_tokens: Maximum number of tokens to use from the dataset for each split in Million. 
                    - The setting of size follows the order of 'train', 'validation', 'test'.
                        (e.g., [3000, 1000] for 3B and 1B tokens respectively for train and validation) 
                    -  The length of the list must match the number of splits in the dataset.
                    - If None or all(x==-1), uses the entire dataset. Applied to each split independently.
                    - Cannnot be set together with max_train_samples and max_eval_samples.
            max_train_samples: Maximum number of training samples to use from the dataset. -1 for all samples.
            max_eval_samples: Maximum number of evaluation samples to use from the dataset. -1 for all samples.

        Returns:
            Processed dataset with the same splits as input.
        """

        # Get column names from the first available split
        if "train" in raw_datasets:
            column_names = list(raw_datasets["train"].features)
        elif "validation" in raw_datasets:
            column_names = list(raw_datasets["validation"].features)
        else:
            # Get the first available split
            first_split = list(raw_datasets.keys())[0]
            column_names = list(raw_datasets[first_split].features)
                
        # Determine which columns to remove during tokenization
        remove_cols = column_names.copy()
        # If specific columns to remove are provided, remove them first
        if columns_to_remove:
            for col in columns_to_remove:
                if col in raw_datasets[list(raw_datasets.keys())[0]].features:
                    raw_datasets = {
                        split: raw_datasets[split].remove_columns([col]) 
                        for split in raw_datasets.keys()
                    }
        
        # 1. Tokenize all splits
        num_proc_to_use = preprocessing_num_workers
        
        if not streaming:
            tokenized_datasets = {}
            for split in raw_datasets.keys():
                logger.info(f"Starting tokenization for {split} dataset (num_proc={num_proc_to_use})")
                tokenized_datasets[split] = raw_datasets[split].map(
                    self._tokenize_function,
                    batched=True,
                    num_proc=num_proc_to_use,
                    remove_columns=[text_column_name],
                    desc=f"Running tokenizer on {split} dataset",
                    load_from_cache_file=not overwrite_cache,
                )
                logger.info(f"Completed tokenization for {split} dataset")
        else:
            tokenized_datasets = {}
            for split in raw_datasets.keys():
                tokenized_datasets[split] = raw_datasets[split].map(
                    self._tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
        
        # 2. Group into fixed size chunks (context + target)
        if not streaming:
            processed_datasets = {}
            for split in tokenized_datasets.keys():
                logger.info(f"Starting grouping for {split} dataset (num_proc={num_proc_to_use})")
                processed_datasets[split] = tokenized_datasets[split].map(
                    self._group_texts,
                    batched=True,
                    num_proc=num_proc_to_use,
                    load_from_cache_file=not overwrite_cache,
                    desc=f"Grouping texts in chunks of {self.total_seq_length} for {split}",
                )
                logger.info(f"Completed grouping for {split} dataset")
        else:
            processed_datasets = {}
            for split in tokenized_datasets.keys():
                processed_datasets[split] = tokenized_datasets[split].map(
                    self._group_texts,
                    batched=True,
                    batch_size=1000,  # Process 1000 docs at a time to form efficient blocks
                )

        ## Calculate the total number of samples
        self._max_samples = None
        if not streaming:
            for split in ['train', 'validation', 'test']:
                if split in processed_datasets:
                    self._max_samples = len(processed_datasets[split])
                    break
        else:
            logger.warning("Streaming datasets are not supported for max_samples calculation. " + \
            "It can be calculated if you set args related to number of samples.")


        # Split into NTP and reconstruction samples
        processed_datasets = self._assign_task_metadata(processed_datasets)

        # 3. Limit the number of tokens to use if max_tokens is specified
        ## Case 1: For max samples
        use_sample_args = False
        if (
            (max_train_samples is not None) and
            (max_eval_samples is not None) and
            (max_train_samples > 0) and
            (max_eval_samples > 0)
        ):
            use_sample_args = True
            if streaming:
                processed_datasets['train'] = processed_datasets['train'].take(max_train_samples)
                processed_datasets['validation'] = processed_datasets['validation'].take(max_eval_samples)
            else:
                processed_datasets['train'] = processed_datasets['train'].select(range(max_train_samples))
                processed_datasets['validation'] = processed_datasets['validation'].select(range(max_eval_samples))
            self._max_samples = max_train_samples if 'train' in processed_datasets else max_eval_samples

        ## Case 2: For max tokens
        if max_tokens is not None and all([x>0 for x in max_tokens]):
            if use_sample_args:
                raise ValueError("max_tokens and max_train_samples/max_eval_samples cannot be set together.")

            if len(max_tokens) != len(list(processed_datasets.keys())):
                raise ValueError(f"The length of max_tokens must match the number of splits in the dataset." + 
                f"Got {len(max_tokens)} but expected {len(list(processed_datasets.keys()))}.")

            splits = ['train', 'validation', 'test'][:len(max_tokens)]
            for split, max_tokens_split in zip(splits, max_tokens):
                # Calculate how many samples we need to reach max_tokens
                # Each sample has total_seq_length tokens
                max_tokens_split = max_tokens_split * 1_000_000 # in million
                max_samples = max_tokens_split // self.total_seq_length
                if streaming:
                    # For streaming datasets, use .take() to limit the number of samples
                    processed_datasets[split] = processed_datasets[split].take(max_samples)
                    num_tokens = max_samples * self.total_seq_length
                    logger.info(f"Limited {split} dataset to {max_samples} samples with {num_tokens/1_000_000_000}B tokens. (This can be not accurate for streaming datasets.)")
                else:
                    # For non-streaming datasets, use .select() to limit the number of samples
                    dataset_len = len(processed_datasets[split])
                    actual_samples = min(max_samples, dataset_len)
                    processed_datasets[split] = processed_datasets[split].select(range(actual_samples))
                    actual_tokens = actual_samples * self.total_seq_length
                    logger.info(f"Limited {split} dataset to {actual_samples} samples. In total actuall {actual_tokens/1_000_000_000}B tokens.")
                    max_samples = actual_samples
                    
                if split == 'train':
                    self._max_samples = max_samples
            
            if self._max_samples is None:
                self._max_samples = max_samples
                logger.info(f"No training split found. Set max samples to {max_samples} according to split {split}.")
        
        if self._max_samples is None:
            raise ValueError("Max samples is not calculated. " + \
            "This typically means that you are using streaming datasets and did not set args related to number of samples.")

        if shuffle_train_set and 'train' in processed_datasets:
            if streaming:
                processed_datasets['train'] = processed_datasets['train'].shuffle(seed=self.seed, buffer_size=10000)
            else:
                processed_datasets['train'] = processed_datasets['train'].shuffle(seed=self.seed)
        return processed_datasets



class _TaskAwareCollatorBase:
    """
    Shared logic for collators that mix NTP and reconstruction samples based on
    per-sample metadata generated by GistDataProcessor.
    """

    def __init__(
        self,
        tokenizer,
        context_length: int = 256,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            tokenizer: The HF tokenizer.
            context_length: Length of text to compress (Input to Encoder).
            prompt: Optional instruction prefix for reconstruction samples.
            NOTE: Important! 
                1. The reconstruction prompt will be placed between the context and the label:
                    [context, prompt, label]
                2. The reconstruction prompt will not be encoded by default.
                3. We add no special tokens to the prompt.
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prompt = prompt or ""
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.prompt_ids = self._encode_prompt(self.prompt)

    def _encode_prompt(self, prompt: str) -> List[int]:
        if not prompt:
            return []
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def _ensure_list(self, input_ids: Union[torch.Tensor, List[int]]) -> List[int]:
        if isinstance(input_ids, torch.Tensor):
            return input_ids.tolist()
        return input_ids

    def _process_ntp_feature(self, feature: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Standard NTP split: first context_length tokens are encoder input, rest are labels."""
        input_ids = self._ensure_list(feature["input_ids"])
        seq_len = len(input_ids)

        if seq_len <= self.context_length:
            raise ValueError("Sequence length must be greater than context length for NTP.")

        pad_id = self.tokenizer.pad_token_id
        labels = input_ids[self.context_length:]
        context_mask = [True] * self.context_length + [False] * (seq_len - self.context_length)
        attention_mask = [0 if token == pad_id else 1 for token in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        }

    def _process_reconstruction_feature(self, feature: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Build reconstruction sample by slicing a segment, prepending prompt, and reusing context/label layout.
        
        We first make sure the context length to be compressed is exactly as "context_length" specified. 
        Thus, after adding the reconstruction prompt, the reconstruction len will be a bit smaller than the context length encoded.

        The prompt will not be encoded by default. 
        The labels will contain labels for all tokens except the context tokens (including the prompt tokens).
            - The labels for the prompt tokens will be assigned to -100.
            - so that in the forward pass, the model only needs to extract the context tokens and leave the rest.

        NOTE: Important! The reconstruction prompt will be placed between the context and the label:
            [context, prompt, label]
        """

        input_ids = self._ensure_list(feature["input_ids"])
        seq_len = len(input_ids)
        target_len = seq_len - self.context_length

        if target_len <= 0:
            raise ValueError("Total sequence length must be greater than context length for reconstruction.")

        # Determine the segment to use based on the segment index
        segment_idx = max(0, feature.get("reconstruction_segment", 0))
        max_start = max(0, seq_len - self.context_length)
        segment_start = min(segment_idx * self.context_length, max_start)
        segment = input_ids[segment_start:segment_start + self.context_length]

        pad_id = self.tokenizer.pad_token_id
        # # A rare case: the segment is shorter than the context length encoded.
        # if len(segment) < self.context_length:
        #     segment = segment + [pad_id] * (self.context_length - len(segment))

        prompt_ids = self.prompt_ids
        prompt_len = len(prompt_ids)

        # Calculate the context budget. 
        # Make sure the context length to be compressed is exactly as "context_length" specified.
        context_tokens = segment[:self.context_length]
        # IMPORTANT: Make sure the context to be encoded are of the same size by padding.
        if len(context_tokens) < self.context_length:
            context_tokens = context_tokens + [pad_id] * (self.context_length - len(context_tokens))

        # Calculate the reconstruction label tokens and apply padding.
        label_len_budget = min(seq_len - self.context_length - prompt_len, self.context_length) # the budget for the label tokens.
        label_tokens = segment[:label_len_budget]
        if len(label_tokens) < (label_len_budget):
            label_tokens = label_tokens + [pad_id] * (label_len_budget - len(label_tokens))

        # Build decoder inputs (prompt + labels) separately from the loss targets
        decoder_input_tokens = prompt_ids + label_tokens

        # Prepare loss targets: ignore prompt tokens and BOS tokens and padding tokens
        labels = decoder_input_tokens.copy()
        labels[:prompt_len] = [-100] * prompt_len
        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None:
            for idx in range(prompt_len, len(labels)):
                if (labels[idx] == bos_id) or (labels[idx] == pad_id):
                    labels[idx] = -100

        # Combine the context part and the decoder input part.
        combined_input = context_tokens + decoder_input_tokens
        attention_mask = [0 if tok == pad_id else 1 for tok in combined_input]
        context_mask = [True] * self.context_length + [False] * (label_len_budget + prompt_len)
        return {
            "input_ids": torch.tensor(combined_input, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        }

    def _postprocess_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return sample

    def _process_feature(self, feature: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        is_ntp = feature.get("for_ntp", True)
        if is_ntp:
            sample = self._process_ntp_feature(feature)
        else:
            sample = self._process_reconstruction_feature(feature)
        return self._postprocess_sample(sample)

    def __call__(self, features):
        if not features:
            raise ValueError("No features provided to the data collator.")
        if "input_ids" not in features[0]:
            raise ValueError("input_ids not found in features, which can mean the features are not tokenized.")

        processed = [self._process_feature(feature) for feature in features] # use for loop because it is fast
        batch = {}
        for key in ["input_ids", "attention_mask", "labels", "context_mask"]:
            batch[key] = torch.stack([sample[key] for sample in processed], dim=0)
        return batch


class DataCollatorForHierarchicalCompressor(_TaskAwareCollatorBase):
    name = 'hier'

    def __init__(
        self,
        tokenizer,
        context_length: int = 256,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        '''
        Args:
            tokenizer: The HF tokenizer.
            context_length: Length of text to compress (Input to Encoder).
            prompt: Optional instruction prefix for reconstruction samples.

        NOTE: The generation length (labels length) will be: total length - context length.
        '''
        super().__init__(
            tokenizer=tokenizer,
            context_length=context_length,
            prompt=prompt,
            **kwargs,
        )


class DataCollatorForICAEFlex(_TaskAwareCollatorBase):
    name = 'icae-flex'

    def __init__(
        self,
        tokenizer,
        context_length: int = 256,
        num_gist_tokens: int = 64,
        gist_token: str = '<gist>',
        prompt: Optional[str] = 'Repeat the previous content:',
        **kwargs,
    ):
        '''
        Args:
            tokenizer: The HF tokenizer.
            context_length: Length of text to compress (Input to Encoder).
            num_gist_tokens: Number of gist tokens to append to the context.
            gist_token: The token to use for the gist tokens.
            prompt: Optional instruction prefix for reconstruction samples.

        NOTE: The generation length (labels length) will be: total length - context length.
        '''
        super().__init__(
            tokenizer=tokenizer,
            context_length=context_length,
            prompt=prompt,
            **kwargs,
        )
        self.num_gist_tokens = num_gist_tokens
        self.gist_token = gist_token
        self.gist_token_id = self.tokenizer.convert_tokens_to_ids(self.gist_token)

    def _postprocess_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Postprocess a sample by inserting gist tokens between context and labels.
        
        For ICAE, the input sequence structure is: [context, gist_tokens, labels]
        where gist tokens act as compressed representations of the context.

        We assume that the context length to be compressed is exactly as "context_length" specified with padding.
        
        Args:
            sample: Dictionary containing 'input_ids', 'attention_mask', and 'labels'
            
        Returns:
            Modified sample with gist tokens inserted and updated masks
        """

        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]

        # Extract context and label portions
        context_part = input_ids[:self.context_length]
        label_part = labels
        context_attention = attention_mask[:self.context_length]
        label_attention = attention_mask[self.context_length:]

        # Create gist tokens to insert between context and labels
        gist_tokens = torch.full(
            (self.num_gist_tokens,),
            self.gist_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        gist_attention = torch.ones(
            (self.num_gist_tokens,),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # Concatenate: [context, gist_tokens, labels]
        new_input_ids = torch.cat([context_part, gist_tokens, label_part], dim=0)
        new_attention_mask = torch.cat([context_attention, gist_attention, label_attention], dim=0)
        
        # Context mask includes both original context and gist tokens
        new_context_mask = torch.zeros_like(new_input_ids, dtype=torch.bool)
        new_context_mask[:self.context_length + self.num_gist_tokens] = True

        sample["input_ids"] = new_input_ids
        sample["attention_mask"] = new_attention_mask.long()
        sample["context_mask"] = new_context_mask
        sample["labels"] = label_part
        return sample



class DataCollatorForICAE(_TaskAwareCollatorBase):
    name = 'icae'

    def __init__(
        self,
        tokenizer,
        context_length: int = 256,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        '''
        Args:
            tokenizer: The HF tokenizer.
            context_length: Length of text to compress (Input to Encoder).
            prompt: Optional instruction prefix for reconstruction samples.

        NOTE: The generation length (labels length) will be: total length - context length.
        '''
        super().__init__(
            tokenizer=tokenizer,
            context_length=context_length,
            prompt=prompt,
            **kwargs,
        )


class DataCollatorFor500x(_TaskAwareCollatorBase):
    name = '500x'

    def __init__(
        self,
        tokenizer,
        context_length: int = 256,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        '''
        Args:
            tokenizer: The HF tokenizer.
            context_length: Length of text to compress (Input to Encoder).
            prompt: Optional instruction prefix for reconstruction samples.

        NOTE: The generation length (labels length) will be: total length - context length.
        '''
        super().__init__(
            tokenizer=tokenizer,
            context_length=context_length,
            prompt=prompt,
            **kwargs,
        )


__all__ = [
    "GistDataProcessor", 
    "DataCollatorForHierarchicalCompressor", 
    "DataCollatorForICAE", 
    "DataCollatorForICAEFlex", 
    "get_data_collator_factory"
    ]




if __name__ == "__main__":
    # 1. Setup Tokenizer (Using Llama-2/3 as example)
    model_id = "meta-llama/Llama-2-7b-hf" 
    # Ensure you have access or swap to "gpt2" for testing
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Settings
    CONTEXT_LEN = 256
    GEN_LEN = 256
    model_structure = DataCollatorForHierarchicalCompressor.name
    
    # 3. Prepare Data Processor
    processor = GistDataProcessor(
        tokenizer=tokenizer, 
        context_length=CONTEXT_LEN, 
        generation_length=GEN_LEN
    )
    
    # 4. Get the iterable dataset
    train_dataset = processor.get_dataset()
    
    # 5. Prepare Collator
    collator = get_data_collator_factory(model_structure)(tokenizer, context_length=CONTEXT_LEN)

    # 6. Simulation: Fetch one batch to verify shapes
    print("Fetching a batch for verification...")
    from torch.utils.data import DataLoader
    
    # Create a simple dataloader
    dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collator)
    
    # Grab first batch
    batch = next(iter(dataloader))
    
    print(f"Encoder Inputs Shape: {batch['encoder_input_ids'].shape}") 
    # Expected: [4, 256]
    
    print(f"Labels Shape:         {batch['labels'].shape}") 
    # Expected: [4, 256]
    
    print("Verification Successful.")

    # 7. How to pass to HuggingFace Trainer
    # from transformers import Trainer, TrainingArguments
    # 
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     max_steps=10000, # Control 3B tokens here via steps * batch_size * seq_len
    #     per_device_train_batch_size=8,
    #     logging_steps=100,
    #     ...
    # )
    # 
    # trainer = Trainer(
    #     model=my_custom_gist_model,
    #     args=training_args,
    #     train_dataset=train_dataset, # Pass the iterable dataset here
    #     data_collator=collator,
    # )
    # trainer.train()