from transformers import TrainerCallback
from loguru import logger
import torch

from src.device_utils import get_device_stats


class LoguruCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # This captures the logs that Trainer produces (loss, lr, etc.)
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")


class DeviceUsageCallback(TrainerCallback):
    def __init__(self, rank:int, logging_steps=50):
            self.logging_steps = logging_steps

    def on_step_end(self, args, state, control, **kwargs):
        # execute this check every N steps
        if state.global_step % self.logging_steps == 0:
            # FIX: Use local_process_index instead of process_index (global rank)
            # Global rank (0-7) causes index out of range on multi-node setups
            stats = get_device_stats(args.local_process_index)
            if stats:
                logger.info(f"Step:{state.global_step}. Rank:{args.process_index} - Device Memory: {stats}")

# Alias for backward compatibility
XPUUsageCallback = DeviceUsageCallback

__all__ = ["LoguruCallback", "DeviceUsageCallback", "XPUUsageCallback"]