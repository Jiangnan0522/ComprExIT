from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.distributed as dist
from transformers import Trainer
from loguru import logger
from traceback import format_exc

MAX_QUESTION_LENGTH = 64
EVAL_SPLIT = 'validation'
MAX_NEW_TOKENS = 32
EVAL_BATCH_SIZE = 128

class CompressInTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Optional SFT evaluation config (passed from train.py).
        # Pop BEFORE calling super() to avoid unexpected-kwarg errors in HF Trainer.
        self.sft_dataset_names: Optional[list[str]] = kwargs.pop("sft_dataset_names", None)
        self.sft_datasets_dir: Optional[str] = kwargs.pop("sft_datasets_dir", None)
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.
        
        Subclassing to average loss across ranks in DDP so that the reported loss
        is the global average, not just the local rank's loss.
        """

        # Average loss across number of accumulation steps and number of ranks in DDP
        # Check distributed status directly instead of relying on args
        dist_ok = dist.is_available() and dist.is_initialized()
        if ("loss" in logs) and dist_ok:
            # Average the loss across all ranks
            # We create a tensor on the correct device to use all_reduce
            with torch.no_grad():
                # logs["loss"] is a float, wrap it in a tensor
                loss_tensor = torch.tensor(logs["loss"], device=self.args.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                
                # Normalize by actual world size
                loss_tensor = loss_tensor / self.accelerator.num_processes
                
                # Normalize by accumulation steps to get per-step loss
                if self.args.gradient_accumulation_steps > 1:
                    loss_tensor = loss_tensor / self.args.gradient_accumulation_steps
                    
                logs["loss"] = loss_tensor.item()

        return super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if self.args.mode == 'ntp':
            return super().evaluate(*args, **kwargs)
        elif self.args.mode == 'sft':
            try:
                from src.evaluation.eval_datasets import Evaluator
                from src.data_processing.sft_datasets import get_sft_dataset_class_factory
                from src.model.inference import (
                    compressing_predict_with_question_and_context,
                    beacon_predict,
                    base_model_predict,
                )
            except Exception as e:
                logger.warning(f"Failed to import evaluation utilities; skipping SFT extra metrics. Error: {e}")
                return {}

            def _parse_result_str(result_str: str) -> dict[str, float]:
                out: dict[str, float] = {}
                for line in (result_str or "").splitlines():
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        out[k] = float(v)
                    except Exception:
                        continue
                return out

            device = getattr(self.args, "device", None)
            if device is None:
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    device = 'xpu'
                elif hasattr(torch, 'cuda') and torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'
                logger.info(f"Using device for evaluation: {device}")
            tokenizer = self.model.tokenizer 

            # Heuristics: reuse training/eval lengths if present in config, otherwise fall back to eval defaults.
            max_context_length = int(getattr(getattr(self.model, "config", None), "context_length", 512))
            # For QA-style evaluation, answers tend to be short.
            max_new_tokens = MAX_NEW_TOKENS
            max_question_length = MAX_QUESTION_LENGTH
            compress_ratio = int(getattr(getattr(self.model, "config", None), "compression_ratio", 4))
            batch_size = EVAL_BATCH_SIZE

            metrics = {}
            for dataset_name in self.sft_dataset_names:
                try:
                    dataset_class = get_sft_dataset_class_factory(dataset_name)
                    dataset = dataset_class(self.sft_datasets_dir)
                    evaluator = Evaluator(
                        dataset, 
                        split=EVAL_SPLIT,
                        max_new_tokens=max_new_tokens,
                        max_question_length=max_question_length,
                        max_context_length=max_context_length,
                        )

                    if hasattr(self.model, '_enable_beacon'):
                        self.model._enable_beacon = True
                        predict_func = beacon_predict
                    elif getattr(self.args, "sft_base_model", False):
                        # Prompt-tuning / plain HF causal LMs (no compression).
                        predict_func = base_model_predict
                    else:
                        predict_func = compressing_predict_with_question_and_context

                    with self.accelerator.autocast():
                        if not self.args.sft_base_model:
                            compress = True
                        else:
                            compress = False
                        # keep autocast enabled for the evaluation during training
                        result_str, _ = evaluator.evaluate(
                            model=self.model,
                            tokenizer=tokenizer,
                            device=device,
                            batch_size=batch_size,
                            compress=compress,
                            with_context=True,
                            compress_ratio=compress_ratio,
                            predict_func=predict_func,
                        )

                    parsed = _parse_result_str(result_str)
                    for k, v in parsed.items():
                        metrics[f"{dataset_name}_{k}"] = v
                except Exception:
                    logger.warning(f"Failed to run SFT extra eval on {dataset_name}; skipping. Error: {format_exc()}")

            logger.info(f"Evaluation Results: {metrics}")
            return metrics
            
        else:
            raise ValueError(f'Invalid mode: {self.args.mode}')


__all__ = ["CompressInTrainer"]
