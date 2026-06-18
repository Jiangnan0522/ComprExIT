from __future__ import annotations

import os

import torch
from loguru import logger
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device_utils import supports_flash_attention_2
from src.model import get_model_factory_from_config


def get_model_and_tokenizer(
    model_folder: str,
    device: str = "cpu",
    attn_implementation: str = "eager",
):
    """
    Load a model + tokenizer from a local checkpoint folder.

    Supports:
    - PEFT prompt tuning adapters (folder contains 'prompt_tuning')
    - Local config-driven models (non-beacon)
    - Activation-beacon models (folder contains 'beacon')
    """
    if "prompt_tuning" in model_folder.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_folder,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        peft_cfg = PeftConfig.from_pretrained(model_folder)
        base_name = peft_cfg.base_model_name_or_path

        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        base_model_embedding_size = base_model.get_input_embeddings().weight.shape[0]
        if base_model_embedding_size < len(tokenizer):
            logger.warning(
                f"Base model embeddings shape {base_model_embedding_size} does not match tokenizer length {len(tokenizer)}. "
                "Resizing tokenizer..."
            )
            base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(base_model, model_folder).to(device)
        model.eval()

        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        logger.info(f"Prompt tuning (PEFT) adapter loaded from {model_folder} (base={base_name})...")
        return model, tokenizer

    if "beacon" not in model_folder:
        logger.info(f"Loading local model from {model_folder}...")
        # local models
        tokenizer = AutoTokenizer.from_pretrained(model_folder)
        model_factory = get_model_factory_from_config(os.path.join(model_folder, "config.json"))
        model_class, model_config_class = model_factory["class"], model_factory["config"]
        model_config = model_config_class.from_pretrained(model_folder)
        model_config.attn_implementation = attn_implementation
        model = model_class.from_pretrained(
            model_folder,
            config=model_config,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            tokenizer=tokenizer,
            attn_implementation=attn_implementation if supports_flash_attention_2() else "eager",
        ).to(device)

        if attn_implementation == "eager" or not supports_flash_attention_2():
            model.set_attn_implementation("eager")

        model.eval()
        return model, tokenizer

    if "beacon" in model_folder:
        # add this if for clarity
        
        logger.info(f"Loading activation beacon model from {model_folder}...")
        # Handle the activation beacon model
        from src.baselines.activation_beacon.src.llama.modeling_llama import (
            LlamaConfig,
            LlamaForCausalLM,
        )

        config = LlamaConfig.from_pretrained(model_folder)
        config.pad_token_id = None

        tokenizer = AutoTokenizer.from_pretrained(
            model_folder,
            trust_remote_code=True,
            # Transformers warns this is needed to avoid incorrect tokenization in some environments.
            fix_mistral_regex=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = LlamaForCausalLM.from_pretrained(
            model_folder,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)

        model._enable_beacon = True
        model.eval()
        return model, tokenizer
