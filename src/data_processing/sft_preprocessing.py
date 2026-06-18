from __future__ import annotations
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from loguru import logger

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .prompte_template import QATemplate
from model.modelling_utils import move_padding_to


SFTMode = Literal["question-aware", "question-unaware"]
TruncSide = Literal["left", "right"]
PAD_TOKEN = "<pad>"


class SFTDataProcessorBase:
    """Preprocessing for SFT (QA-style supervision).

    Output contract (per example), when `pad_to_max_length=True` (default):
        - input_ids: length == max_context_length + max_generation_length
        - attention_mask: same length
        - context_mask: same length; True for first max_context_length positions
        - labels: length == max_generation_length; -100 for non-loss positions

    Mode:
        - question-aware: compressor sees context; query contains question+gold; loss on gold only.
        - question-unaware: compressor sees context; query contains question+gold; question tokens are masked (-100), loss on gold only.

    Note: If in the future you want \"question in context\" compression, add a mode that moves question tokens
    into the context region (context_mask=True) and removes them from labels entirely.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int = 512,
        max_generation_length: int = 32,
        seed: int = 42,
        **kwargs: Any,
    ):
        self.tokenizer = tokenizer
        self.max_context_length = int(max_context_length)
        self.max_generation_length = int(max_generation_length)
        self.seed = seed

        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set before preprocessing.")

    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        raise NotImplementedError



class SFTDataProcessor(SFTDataProcessorBase):
    """Concrete SFT processor emitting fixed-length fields for CompressIn training."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int = 512,
        max_generation_length: int = 128,
        mode: SFTMode = "question-unaware",
        template: Optional[QATemplate] = None,
        # Truncation behavior
        truncate_context_from: TruncSide = "left",
        append_eos_to_answer: bool = True, # append by default
        seed: int = 42,
        **kwargs: Any,
    ):
        """
        Args:
            max_context_length: The maximum length of the context to be compressed.
            max_generation_length: The maximum length of the generation (including the prefix and the answer).
            mode: The mode of the SFT. Can be "question-aware" or "question-unaware".
        """
        super().__init__(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            max_generation_length=max_generation_length,
            seed=seed,
            **kwargs,
        )
        self.mode: SFTMode = mode
        self.template = template or QATemplate()
        self.truncate_context_from: TruncSide = truncate_context_from
        self.append_eos_to_answer = bool(append_eos_to_answer)

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            raise ValueError("tokenizer.pad_token_id and tokenizer.eos_token_id must be different because we will learn to generate eos tokens.")

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _truncate_and_pad_ids(self, ids: List[int], length: int, trunc_from: TruncSide) -> List[int]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set before preprocessing.")

        if len(ids) > length:
            if trunc_from == "left":
                ids = ids[-length:]
            else:
                ids = ids[:length]
        if len(ids) < length:
            ids = ids + [pad_id] * (length - len(ids))
        return ids

    def _build_texts(self, example: Dict[str, Any]) -> Dict[str, str]:
        # Expected fields (QA-style)
        missing = [k for k in ("context", "question", "gold") if k not in example]
        if missing:
            raise KeyError(
                "SFT preprocessing expects dataset columns: 'context', 'question', 'gold'. "
                f"Missing: {missing}. Available keys: {list(example.keys())}"
            )

        context = example["context"]
        question = example["question"]
        answer = example["gold"]

        context_text = self.template.build_context_text(context)
        prefix_text = self.template.build_prefix_text(question)
        answer_text = self.template.build_answer_text(answer)
        return {"context_text": context_text, "prefix_text": prefix_text, "answer_text": answer_text}

    def _encode_query_and_labels(self, prefix_text: str, answer_text: str) -> Dict[str, List[int]]:
        """Build fixed-length query tokens and query-only labels (answer supervised, prefix masked).

        query means the input part that is not compressed, including the prefix(no-gradient) and the answer.
        """

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set before preprocessing.")

        prefix_ids = self._encode(prefix_text)
        answer_ids = self._encode(answer_text)

        if self.append_eos_to_answer and self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        max_len = self.max_generation_length

        # Preserve answer tokens as much as possible by truncating prefix first.
        if len(answer_ids) >= max_len:
            # Answer alone fills the budget; drop prefix entirely.
            answer_ids = answer_ids[-max_len:]
            query_ids = answer_ids
            labels = answer_ids.copy()
            prefix_len_in_query = 0
        else:
            prefix_budget = max_len - len(answer_ids)
            prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget > 0 else []
            query_ids = prefix_ids + answer_ids
            prefix_len_in_query = len(prefix_ids)
            # Important! Mask prefix tokens from loss. Only calculate loss on answer tokens.
            labels = ([-100] * prefix_len_in_query) + answer_ids

        # Pad to fixed length (labels padded with -100)
        if len(query_ids) < max_len:
            query_ids = query_ids + [pad_id] * (max_len - len(query_ids))
            labels = labels + ([-100] * (max_len - len(labels)))
        else:
            query_ids = query_ids[:max_len]
            labels = labels[:max_len]

        # Mask padding tokens from loss. We do not mask BOS.
        bos_id = self.tokenizer.bos_token_id
        for i, tok in enumerate(query_ids):
            if tok == pad_id:
                labels[i] = -100
        return {"query_ids": query_ids, "labels": labels}

    def _encode_sft_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        texts = self._build_texts(example)

        # Pre-truncate at character level to avoid tokenizing extremely long documents
        # (e.g. TriviaQA contexts can exceed 100k tokens). Use ~6 chars/token as a
        # generous estimate; exact token-level truncation still happens below.
        max_chars = self.max_context_length * 6
        context_text = texts["context_text"]
        if len(context_text) > max_chars:
            if self.truncate_context_from == "left":
                context_text = context_text[-max_chars:]
            else:
                context_text = context_text[:max_chars]

        context_ids = self._encode(context_text)
        context_ids = self._truncate_and_pad_ids(
            context_ids, self.max_context_length, trunc_from=self.truncate_context_from
        )

        query_and_labels = self._encode_query_and_labels(
            prefix_text=texts["prefix_text"], answer_text=texts["answer_text"]
        )

        input_ids = context_ids + query_and_labels["query_ids"]
        pad_id = self.tokenizer.pad_token_id
        attention_mask = [0 if tok == pad_id else 1 for tok in input_ids]
        context_mask = ([True] * self.max_context_length) + ([False] * self.max_generation_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
            "labels": query_and_labels["labels"],
        }

    def _batch_encode_sft_example(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Process a batch of examples. 
        
        When batched=True, the input is a dict of lists (one list per column).
        We need to return a dict of lists for the output columns.
        """
        batch_size = len(examples["context"])
        
        results = {
            "input_ids": [],
            "attention_mask": [],
            "context_mask": [],
            "labels": []
        }
        
        # Process each example in the batch
        for i in range(batch_size):
            # Extract single example from batch
            example = {key: examples[key][i] for key in examples.keys()}
            
            # Process using the single-example method
            processed = self._encode_sft_example(example)
            
            # Append results
            results["input_ids"].append(processed["input_ids"])
            results["attention_mask"].append(processed["attention_mask"])
            results["context_mask"].append(processed["context_mask"])
            results["labels"].append(processed["labels"])
        
        return results

    def process_dataset(
        self,
        dataset: DatasetDict,
        shuffle_train_set: bool = True,
        keep_columns: Optional[List[str]] = None,
        overwrite_cache: bool = False,
    ) -> DatasetDict:
        """Map over each split and emit fixed-length tensors as python lists.

        By default, removes all original columns from the raw dataset and keeps only:
        `input_ids`, `attention_mask`, `context_mask`, `labels`.

        Dataset format of a single record:
            {
                "context": str,
                "question": str,
                "gold": str,
            }
        """

        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict, got {type(dataset)}")

        processed: Dict[str, Dataset] = {}
        for split, ds in dataset.items():
            remove_cols = list(ds.column_names)

            if keep_columns:
                remove_cols = [c for c in remove_cols if c not in set(keep_columns)]
        
            if shuffle_train_set and split == "train":
                ds = ds.shuffle(seed=self.seed)

            processed[split] = ds.map(
                self._batch_encode_sft_example,
                num_proc=min(8, os.cpu_count()),
                batched=True,
                batch_size=500,
                load_from_cache_file=not overwrite_cache,
                remove_columns=remove_cols,
                desc=f"SFT preprocessing ({split})",
            )

        return DatasetDict(processed)


class DataCollatorForSFT:
    """Minimal collator to stack fixed-length SFT examples into torch tensors."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("No features provided to DataCollatorForSFT.")

        batch: Dict[str, torch.Tensor] = {}
        for key, dtype in [
            ("input_ids", torch.long),
            ("attention_mask", torch.long),
            ("labels", torch.long),
            ("context_mask", torch.bool),
        ]:
            batch[key] = torch.tensor([f[key] for f in features], dtype=dtype)
        return batch


class DataCollatorForSFTBaseModel:
    """Collator for SFT base model(no-compression) finetuning."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # all examples should have the same context length(including context paddings)
        context_length = sum(features[0]['context_mask'])
        features_processed = []
        for example in features:
            example_processed = deepcopy(example)
            example_processed['labels'] = self._to_full_labels(example_processed, context_length)
            features_processed.append(example_processed)

        batch: Dict[str, torch.Tensor] = {}
        for key, dtype in [
            ("input_ids", torch.long),
            ("attention_mask", torch.long),
            ("labels", torch.long),
            # remove context_mask key from batch
        ]:
            batch[key] = torch.tensor([f[key] for f in features_processed], dtype=dtype)

        # move paddings to the right (so that the prefix virtual tokens are at the left)
        batch["input_ids"], batch["attention_mask"], batch["labels"] = move_padding_to(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
            padding_side="right",
        )
        return batch

    # Convert query-only labels -> full-length labels aligned to input_ids
    def _to_full_labels(self, example: dict, context_length: int) -> Dict[str, Any]:
        return ([-100] * context_length) + example["labels"]


__all__ = [
    "SFTDataProcessorBase",
    "SFTDataProcessor",
    "DataCollatorForSFT",
]


if __name__ == "__main__":
    PAD_TOKEN = "<pad>"
    # Demo/sanity run: pass the base model (local path or HF id) as the first argument.
    tokenizer_path = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"additional_special_tokens": [PAD_TOKEN]})
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    processor = SFTDataProcessor(tokenizer=tokenizer)
    dataset = Dataset.from_dict({
        "context": ["The context is about the question and the answer.", "The context is about the question and the answer."],
        "question": ["What is the question?", "What is the question?"],
        "gold": ["The answer is the answer.", "The answer is the answer."],
    })
    dataset = DatasetDict({"train": dataset, "validation": dataset})
    processed_dataset = processor.process_dataset(dataset)

    collator = DataCollatorForSFT()
    batch = collator([processed_dataset["train"][0], processed_dataset["train"][1]])
    import pdb; pdb.set_trace()