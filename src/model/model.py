from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from loguru import logger
from abc import ABC, abstractmethod
from overrides import overrides

import torch
import torch.nn.functional as F
from torch import nn

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    CONFIG_MAPPING,
    PreTrainedTokenizer,
    DynamicCache,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import get_peft_model, LoraConfig

from src.model.pooling import get_pooling_factory
from src.model.modelling_utils import move_padding_to
from src.device_utils import get_device_module, supports_flash_attention_2

DEVICE_MODULE, DEVICE_TYPE = get_device_module()


def get_model_factory(model_structure: str):
    """
    Get the model class and config class for the given model structure.
    """
    if model_structure == "hier":
        return {'class':HierarchicalCompressor, 'config':HierarchicalCompressorConfig}
    elif model_structure == "icae-flex":
        return {'class':ICAEFlex, 'config':ICAEFlexConfig}
    elif model_structure == "icae":
        return {'class':ICAE, 'config':ICAEConfig}
    elif model_structure == "500x":
        return {'class':FiveHundredX, 'config':FiveHundredXConfig}
    elif model_structure == "sac":
        return {'class':SAC, 'config':SACConfig}
    else:
        raise ValueError(f"Invalid model structure: {model_structure}")


def get_model_factory_from_config(config_path: str):
    """
    Get the model class and config class from a config file.
    """
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    architecture = config_dict.get("architectures", [None])[0]
    
    model_class = globals().get(architecture)
    config_class = globals().get(f"{architecture}Config")
    
    if model_class and config_class:
        return {'class': model_class, 'config': config_class}
    else:
        raise ValueError(f"Unknown architecture in config: {architecture}")


class EncoderDecoderCompressorBaseConfig(PretrainedConfig):
    model_type = "end2end_compression"
    # for base model
    lm_name_or_path: Optional[str] = None
    attn_implementation: str = 'flash_attention_2' # ["flash_attention_2", "sdpa", "eager"]
    dtype: Optional[str] = 'float32'
    # for compressor
    top_k_layers: int = -1
    context_length: int = 256
    training_freezing_mode: str
    # for projection mlp
    num_hidden_layers: int = 2
    projector_gain: float = 0.4
    # for lora
    lora_compressor: bool = False # whether to apply lora to the compressor
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_target_modules: Optional[List[str]] = None
    # backbone info
    backbone_config_dict: Optional[Dict[str, Any]] = None


class HierarchicalCompressorConfig(EncoderDecoderCompressorBaseConfig):
    model_type = "hierarchical_compressor"
    # for compressor
    pooling_method: str = "sliding" # ["avg", "last", "sliding", "echo"]
    ### for sliding window pooling
    add_global_avg: bool = False
    # --- for TokenLevelLayerwisePooling ("token-level-layerwise") ---
    layerwise_pooling_layers: Optional[List[int]] = None
    layerwise_pooling_gate_hidden: int = 256 # hidden size for the gating MLP
    layerwise_pooling_temperature: float = 0.7 # temperature for the softmax function
    chunk_attn_hidden_size: int = 256 # hidden size for the chunk attention
    chunk_attn_num_heads: int = 4 # number of heads for chunk attention
    # --- For DynamicConcatenatedLayerwiseChunkAttentionPooling ("dy-concat-mh") ---
    reduced_hidden_size: int = 64 # hidden size for the reduced hidden states
    # --- For OptimalTransportPooling ("ot") ---
    ot_window_size: int = 128 # window size for the OT pooling
    ot_n_iter: int = 30 # number of iterations for the Sinkhorn algorithm
    ot_metric_dim: int = 256 # hidden size for the metric embeddings
    ## OT Ablations
    ab_ot_shuffle_anchors: bool = False # whether to shuffle the anchor order
    # --- For Map-Reduce ---
    map_reduce_seg_len: int = 512 # segment length for the map-reduce compression
    trained_with_map_reduce: bool = False # whether the model was trained with map-reduce

class ICAEFlexConfig(HierarchicalCompressorConfig):
    model_type = "icae-flex"
    gist_token:str='<gist>'
    # for lora
    lora_compressor:bool=True # lora default to be True for ICAE
    lora_r:int=128
    lora_alpha:int=32
    lora_dropout:float=0.05
    lora_bias:str="none"
    lora_task_type:str="CAUSAL_LM"
    lora_target_modules:List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]


class ICAEConfig(HierarchicalCompressorConfig):
    model_type = "icae"
    num_memory_tokens: int = 64
    # ICAE typically uses 0 projection layers (direct connection)
    num_hidden_layers: int = 0
    # for lora
    lora_compressor: bool = True
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]


class FiveHundredXConfig(EncoderDecoderCompressorBaseConfig):
    model_type = "500x"
    num_memory_tokens: int = 64
    # FiveHundredX specific defaults
    lora_compressor: bool = True 
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["q_proj", "v_proj"]
    # For PartialLLMModel to use all layers
    top_k_layers: int = -1 
    

class SACConfig(EncoderDecoderCompressorBaseConfig):
    model_type = "sac"
    mem_size: int = 128                # compressed tokens per chunk
    compress_ratio: int = 4          # subsampling stride
    chunk_size: int = 512           # tokens per encoder chunk
    # LoRA equivalent to SAC's custom LoRA (rank=128, scale=2)
    lora_compressor: bool = True
    lora_r: int = 128
    lora_alpha: int = 256             # scale = alpha/r = 2.0, matching SAC's hardcoded scale
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = ["q_proj", "v_proj"]
    top_k_layers: int = -1            # all layers needed for full KV cache
    num_hidden_layers: int = 0        # no projector MLP (KV cache passed directly)
    attn_implementation: str = 'eager' # bidirectional attention needs sdpa (not flash_attention_2)


class PartialLLMModel(nn.Module):
    """
    Clones the backbone of a causal LLM and retains only the first ``top_k_layers`` Transformer blocks. The resulting
    module behaves like a standalone model that begins from the token embeddings and halts after the selected depth.
    """

    def __init__(self, config: EncoderDecoderCompressorBaseConfig, lm: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        """
        Args:
            config: The configuration of the model.
                - top_k_layers: The number of layers to keep from the base model. -1 for using all layers.
            lm: The base model.
                - The base model is a causal LLM model.
            tokenizer: The tokenizer of the base model.
        """
        self.config = config
        self.top_k_layers = config.top_k_layers
        self.tokenizer = tokenizer

        # Record the base model type for future reference.
        if not hasattr(self.config, 'lm_type') or self.config.lm_type is None:
            self.config.lm_type = getattr(lm.config, "model_type", None)

        # Validate that the requested truncation depth is not greater than the number of layers in the base model.
        num_layers = getattr(lm.config, "num_hidden_layers", None)
        if num_layers is not None and self.top_k_layers > num_layers:
            raise ValueError(
                f"Requested top_k_layers={self.top_k_layers}, but base model only has {num_layers} layers."
            )

        backbone = self._extract_backbone(lm)
        self.backbone = self._build_partial_backbone(backbone, self.top_k_layers)
        del lm

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None, inputs_embeds: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Runs the underlying model and returns the hidden state generated after the ``top_k_layers``-th block.
        """
        if inputs_embeds is not None:
             return self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return self.backbone(input_ids, attention_mask, **kwargs)

    def _extract_backbone(self, lm: PreTrainedModel) -> PreTrainedModel:
        """
        Returns the autoregressive backbone (e.g. `LlamaModel`) underneath an AutoModelForCausalLM wrapper.
        """

        for attribute in ("model", "transformer", "language_model"):
            if hasattr(lm, attribute):
                return getattr(lm, attribute)
        raise ValueError("Unable to locate backbone inside the supplied language model.")

    def _build_partial_backbone(self, backbone: PreTrainedModel, top_k_layers: int) -> PreTrainedModel:
        """
        Creates a copy of the provided backbone that only keeps the first `top_k_layers` by:
            - Initialise a new k-layer LLM from the backbone's configuration.
            - Load the weights from the backbone into the new model.
            - Return the new model.
        """

        if top_k_layers == -1:
            if (not self.config.lora_compressor) and (self.config.training_freezing_mode == 'compress+llm'):
                logger.info(f"[PartialLLMModel] Detected [LoRA] NOT applied and freezing [compress+llm]. Sharing the backbone with the decoder to save memory: {backbone.__class__.__name__}.")
                return backbone

            logger.info(f"[PartialLLMModel] Using all layers from the backbone by cloning it: {backbone.__class__.__name__}.")
            cloned = backbone.__class__(backbone.config)
            cloned.load_state_dict(backbone.state_dict())
            return cloned

        logger.info(f"[PartialLLMModel] Building partial backbone with {top_k_layers} layers.")

        if not hasattr(backbone, "config"):
            raise ValueError("Backbone missing configuration object, cannot build partial model.")

        backbone_layer_list = getattr(backbone, "layers", None)
        if not isinstance(backbone_layer_list, nn.ModuleList):
            raise ValueError(
                "Unsupported backbone architecture: expected a `layers` ModuleList attribute. "
                "Currently supported models include LLaMA, Qwen, and Mistral families."
            )
        # Modify the layer-num config of the backbone model
        truncated_config_dict = backbone.config.to_dict()
        truncated_config_dict["num_hidden_layers"] = top_k_layers
        
        # FIX: Also truncate layer_types if present (Qwen3 specific)
        if "layer_types" in truncated_config_dict and isinstance(truncated_config_dict["layer_types"], list):
             truncated_config_dict["layer_types"] = truncated_config_dict["layer_types"][:top_k_layers]

        # Instantiate a new k-layer LLM from the new configuration.
        ## The model will only contain the embedding layer and the first top_k_layers attention layers with no prediction head.
        logger.info(f"  - Instantiating a new {top_k_layers}-layer LLM from the new configuration.")
        truncated_config = backbone.config.__class__(**truncated_config_dict)
        partial_backbone = backbone.__class__(truncated_config)
        self.config.backbone_config_dict = truncated_config_dict # update the backbone config after layer truncation

        # Load the weights from the backbone into the new model.
        logger.info(f"  - Loading weights from the backbone into the new model.")
        state_dict = backbone.state_dict()
        trimmed_state_dict = self._trim_state_dict(state_dict, top_k_layers)
        partial_backbone.load_state_dict(trimmed_state_dict, strict=False)

        logger.info(f"Partial backbone built successfully.")
        return partial_backbone

    def _trim_state_dict(self, state_dict: Dict[str, torch.Tensor], top_k_layers: int) -> Dict[str, torch.Tensor]:
        """
        Drops weights belonging to layers beyond the retained prefix to avoid loading unnecessary parameters.
        """

        trimmed: Dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            layer_index = self._extract_layer_index(key)
            # drop further layers and non-attention-layer layers
            if (layer_index is not None) and (layer_index >= top_k_layers):
                continue
            trimmed[key] = value

        return trimmed

    def _extract_layer_index(self, key: str) -> Optional[int]:
        """
        Parses a state-dict key and returns the layer index if it is an attention layer, otherwise return None.
        """

        layer_markers = [
            "layers.",
            "h.",
            "decoder.layers.",
            "encoder.layers.",
        ]
        for marker in layer_markers:
            if marker in key:
                suffix = key.split(marker, 1)[1]
                if "." not in suffix:
                    continue
                potential_index = suffix.split(".", 1)[0]
                if potential_index.isdigit():
                    return int(potential_index)
        return None

    def _qualname(self, cls: type) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    def _import_class(self, path: Optional[str]):
        if not path:
            raise ValueError("Class path is required to instantiate backbone.")
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs):
        return self.backbone.resize_token_embeddings(new_num_tokens, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """A dummy function for LoRA's "CAUSAL_LM" task type.
        Generation won't be performed for this PartialLLMModel.
        """
        return {"input_ids": input_ids, **kwargs}



class ProjectionMLP(nn.Module):
    def __init__(self, config: EncoderDecoderCompressorBaseConfig):
        super().__init__()
        self.config = config
        layers = []

        if config.num_hidden_layers > 0:
            for _ in range(config.num_hidden_layers - 1):
                layers.append(nn.Linear(config.hidden_size, config.hidden_size))
                layers.append(nn.GELU())
            layers.append(nn.Linear(config.hidden_size, config.hidden_size)) # no activation for the last layer
            self.mlp = nn.Sequential(*layers)
            self.initialize_weights()
        else:
            logger.info(f"Using identity projection MLP because num_hidden_layers is 0.")
            self.mlp = nn.Identity()
        
        # NEW: Learnable output scaling
        # Initialize to 0.1 so the output is small (0.1 * 1.0 = 0.1 ~ close to 0.02 order of magnitude)
        # But allow it to grow if needed.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multiply by scale
        return self.mlp(x)

    def initialize_weights(self):
        """We need to initialise as close to the llm's embedding distribution as possible"""     
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # Revert to standard Xavier init for healthy internal gradients
                nn.init.xavier_uniform_(module.weight, gain=self.config.projector_gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)



class EncoderDecoderCompressorBase(PreTrainedModel, ABC):
    config_class = EncoderDecoderCompressorBaseConfig
    _supports_flash_attn_2 = supports_flash_attention_2()
    supports_gradient_checkpointing = True

    """
        An encoder-decoder base structure which is composed of:
            - an encoder: a partial LLM model
            - a projector: a MLP to project the compressed tokens to the embedding space of the base model
            - a decoder: a projection MLP

        Remember to make sure padding tokens always lie at the front of the sequence (no interleaving padding).
            - handle the padding tokens' positions when passing the context to the compressor.
            - handle the padding tokens' positions when passing the final sequence to the decoder.

    """

    def __init__(
        self,
        config: EncoderDecoderCompressorBaseConfig,
        from_pretrained: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs
    ):
        super().__init__(config)

        self.config = config
        
        # Tokenizer handling:
        # - Prefer an explicitly provided tokenizer to avoid mismatch with preprocessing-time modifications.
        # - Otherwise, prefer loading from the directory/model-id this config was loaded from (checkpoint),
        #   falling back to the raw backbone model folder/id.
        if tokenizer is not None:
            self.tokenizer = tokenizer
            logger.info(
                f"Using provided tokenizer ({self.tokenizer.__class__.__name__}); "
                "skipping internal AutoTokenizer loading."
            )
        else:
            tokenizer_path = getattr(config, "_name_or_path", None) or config.lm_name_or_path
            # If the config was loaded from a local checkpoint dir that does NOT contain tokenizer files,
            # fall back to the raw backbone tokenizer to avoid breaking older checkpoints.
            if tokenizer_path and os.path.isdir(tokenizer_path):
                has_tokenizer_files = any(
                    os.path.exists(os.path.join(tokenizer_path, f))
                    for f in ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
                )
                if (not has_tokenizer_files) and config.lm_name_or_path and (config.lm_name_or_path != tokenizer_path):
                    logger.warning(
                        f"Tokenizer files not found under config._name_or_path={tokenizer_path}. "
                        f"Falling back to lm_name_or_path={config.lm_name_or_path}. "
                        "If you modified the tokenizer during preprocessing, make sure you save it alongside the model."
                    )
                    tokenizer_path = config.lm_name_or_path

            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # skip the weight initialization of the model by setting from_pretrained=True 
        # (for loading the model from a checkpoint) to speed up the model loading.
        if not from_pretrained:
            # Initialise the architecture of the model.
            # Prefer the backbone config embedded in this checkpoint (``backbone_config_dict``) so
            # the model can be rebuilt without the original base model on disk -- e.g. when loading
            # a checkpoint downloaded from the Hub, where ``lm_name_or_path`` points to a local path
            # that does not exist on the user's machine. Fall back to loading the base model's config
            # from ``lm_name_or_path`` for the first build during training (before the backbone
            # config has been embedded into this config).
            backbone_config_dict = getattr(config, "backbone_config_dict", None)
            if backbone_config_dict:
                lm_config = CONFIG_MAPPING[backbone_config_dict["model_type"]].from_dict(
                    backbone_config_dict
                )
            else:
                lm_config = AutoConfig.from_pretrained(config.lm_name_or_path)

            # Enforce HF attention backend selection in the lm_config.
            ## beacuse simply setting attn_implementation in from_config cannot override the setup in the lm_config passed in.
            lm_config._attn_implementation = config.attn_implementation
            lm_config.attn_implementation = config.attn_implementation

            self.lm = AutoLigerKernelForCausalLM.from_config(lm_config, trust_remote_code=True)
            logger.info(f"Base model attn implementation: {config.attn_implementation}")
            self.compressor = PartialLLMModel(config, lm=self.lm, tokenizer=self.tokenizer)
            
            # FIX: Resize token embeddings if tokenizer size differs from model vocab size
            if len(self.tokenizer) != self.lm.config.vocab_size:
                logger.warning(f"Resizing token embeddings from {self.lm.config.vocab_size} to {len(self.tokenizer)}. This may due to vocab padding for hardware efficiency.")
                self.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            
            if config.lora_compressor:
                target_modules = config.lora_target_modules
                if target_modules is None:
                    # Default to common modules for Llama/Mistral/Qwen if not specified
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    logger.info(f"No lora_target_modules specified. Defaulting to {target_modules}")

                self.compressor = get_peft_model(self.compressor, LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    bias=config.lora_bias,
                    task_type=config.lora_task_type,
                    target_modules=target_modules
                ))
                logger.info(f"LoRA applied to the compressor.")

            config.hidden_size  = self.lm.config.hidden_size
            # Register shared weights if any
            self._register_shared_backbone_weights()

        # Initialise the MLP projector in all cases
        self.projector = ProjectionMLP(config)
        if hasattr(config, 'trained_with_map_reduce') and (config.trained_with_map_reduce):
            logger.info(f"Model was trained with map-reduce. Enabling map-reduce.")
            self.enable_map_reduce()

    @classmethod
    def from_pretrained_submodules(
        cls, 
        config: EncoderDecoderCompressorBaseConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs
    ):
        """
        Initialize model with pretrained lm and compressor built from it.
        We don't need to initialize the model architecture as we are loading the model from an existing checkpoint.
        
        Args:
            lm_name_or_path: Path to pretrained base model
            compressor_config: Config for the compressor (without lm_name_or_path)
            **kwargs: Additional arguments
        """
        
        # Load the pretrained base model
        logger.info(f"Loading base model from {config.lm_name_or_path}.")
        
        # Convert dtype string to torch dtype
        torch_dtype = torch.float32
        if config.dtype:
            dtype_map = {
                "auto": "auto",
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(config.dtype, torch.float32)
        
        lm = AutoLigerKernelForCausalLM.from_pretrained(
            config.lm_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=config.attn_implementation,
            low_cpu_mem_usage=True,
            device_map="cpu"  # Load to CPU first, let Trainer move to XPU
        )
        logger.info(f"Base model loaded.")
        logger.info(f"Base model attn implementation: {config.attn_implementation}")

        # skip the weight initialization of the model by setting from_pretrained=True
        config.hidden_size = lm.config.hidden_size # add the hidden size to the config for the projector
        model = cls(config, from_pretrained=True, tokenizer=tokenizer)

        # Create compressor from the loaded base model
        logger.info(f"Building partial backbone compressor from the base model.")
        compressor = PartialLLMModel(config=config, lm=lm, tokenizer=model.tokenizer) # would be of the same dtype as the base model
        logger.info(f"Compressor built successfully.")

        if config.lora_compressor:
            target_modules = config.lora_target_modules
            if target_modules is None:
                # Default to common modules for Llama/Mistral/Qwen if not specified
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                logger.info(f"No lora_target_modules specified. Defaulting to {target_modules}")

            compressor = get_peft_model(compressor, LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
                task_type=config.lora_task_type,
                target_modules=target_modules
            ))
            logger.info(f"LoRA applied to the compressor.")


        # Replace with pretrained components
        model.compressor = compressor
        model.lm = lm
        
        # Ensure projector is in fp32 - Trainer will handle precision via bf16/fp16 training args
        # Frozen components (compressor, lm) stay in their loaded dtype (bf16 recommended for memory)
        model.projector = model.projector.to(torch.float32)
        
        # Resize token embeddings to match tokenizer (important if subclass added tokens)
        model.resize_token_embeddings(len(model.tokenizer))
        
        logger.info(f"Model dtype configuration:")
        logger.info(f"  - LLM dtype: {next(model.lm.parameters()).dtype}")
        logger.info(f"  - Compressor dtype: {next(model.compressor.parameters()).dtype}")
        if model.config.num_hidden_layers > 0:
            logger.info(f"  - Projector dtype: {next(model.projector.parameters()).dtype}")
        else:
            logger.info(f"  - Projector dtype: no parameters")
        
        # Register shared weights if any
        model._register_shared_backbone_weights()

        return model

    def _register_shared_backbone_weights(self):
        """
        Register shared weights between the compressor's backbone and the LM's backbone
        to prevent errors during checkpoint saving when they share the same underlying module.
        """
        if not hasattr(self.compressor, "backbone"):
            return

        compressor_backbone = self.compressor.backbone
        
        # Find the backbone in self.lm
        lm_backbone = None
        lm_backbone_name = None
        
        # Common backbone attribute names in HF models
        for attribute in ("model", "transformer", "language_model"):
            if hasattr(self.lm, attribute):
                candidate = getattr(self.lm, attribute)
                # Check identity
                if candidate is compressor_backbone:
                    lm_backbone = candidate
                    lm_backbone_name = attribute
                    break
        
        if lm_backbone is None:
            logger.info(f"No shared backbone found between compressor and LM. Skipping tied weights registration.")
            return

        logger.warning(f"Found shared backbone between compressor and LM. Registering tied weights for saving. Please make sure this is what you want.")

        tied_keys = []
        prefix_lm = f"lm.{lm_backbone_name}"
        prefix_compressor = "compressor.backbone"
        
        # Register all parameters
        for name, _ in lm_backbone.named_parameters():
            # name is relative to backbone
            tied_keys.append(f"{prefix_lm}.{name}")
            tied_keys.append(f"{prefix_compressor}.{name}")
            
        # Register all buffers (just in case)
        for name, _ in lm_backbone.named_buffers():
            tied_keys.append(f"{prefix_lm}.{name}")
            tied_keys.append(f"{prefix_compressor}.{name}")

        if not hasattr(self, "_dynamic_tied_weights_keys") or self._dynamic_tied_weights_keys is None:
             self._dynamic_tied_weights_keys = []
             
        # Add to the list
        current_set = set(self._dynamic_tied_weights_keys)
        for k in tied_keys:
            if k not in current_set:
                self._dynamic_tied_weights_keys.append(k)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # disable kv-cache during training when checkpointing is on
        if hasattr(self, "lm") and hasattr(self.lm, "config"):
            self.lm.config.use_cache = False
        if hasattr(self, "compressor") and hasattr(self.compressor, "config"):
            self.compressor.config.use_cache = False

        if hasattr(self, "lm") and hasattr(self.lm, "gradient_checkpointing_enable"):
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self, "compressor") and hasattr(self.compressor, "gradient_checkpointing_enable"):
            self.compressor.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self, "lm") and hasattr(self.lm, "gradient_checkpointing_disable"):
            self.lm.gradient_checkpointing_disable()
        if hasattr(self, "compressor") and hasattr(self.compressor, "gradient_checkpointing_disable"):
            self.compressor.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.compressor.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs):
        output = self.lm.resize_token_embeddings(new_num_tokens, **kwargs)
        self.compressor.resize_token_embeddings(new_num_tokens, **kwargs)
        return output

    def _unfreeze_lora_adapters(self, module: nn.Module, module_name: str = "module") -> int:
        """
        Best-effort unfreeze for PEFT/LoRA parameters inside a module that might have just been frozen.

        Returns:
            Number of parameters (tensors) that were set to requires_grad=True.
        """
        if module is None:
            return 0

        # If this is a PEFT model, this will flip adapter layers back on.
        # (Safe to call conditionally; plain HF modules won't have it.)
        if hasattr(module, "enable_adapter_layers") and callable(getattr(module, "enable_adapter_layers")):
            try:
                module.enable_adapter_layers()
            except Exception as e:
                logger.warning(f"Failed to enable_adapter_layers() on {module_name}: {type(e).__name__}: {e}")

        # Fallback: unfreeze by name pattern for LoRA params.
        # PEFT LoRA commonly names params with 'lora_' or 'lora_A'/'lora_B'. It may also store extra trainables under
        # 'modules_to_save'.
        enabled = 0
        for n, p in module.named_parameters(recurse=True):
            if not isinstance(p, torch.nn.Parameter):
                continue
            if (
                ("lora_" in n)
                or ("lora_A" in n)
                or ("lora_B" in n)
                or ("lora_embedding" in n)
                or ("modules_to_save" in n)
            ):
                if not p.requires_grad:
                    p.requires_grad = True
                    enabled += 1
        if enabled > 0:
            logger.info(f"Unfroze {enabled} LoRA/adapter parameter tensors inside {module_name}.")
        return enabled

    def stop_gradient(self, mode: str = 'compress+llm', unfreeze_lora: bool = False):
        """
        Stop gradients for different components based on the training mode.
        
        Args:
            mode: Training mode that determines which components to freeze
                - 'compress+llm' or 'both': Freeze both compressor and LLM, train only projector
                - 'compress': Freeze compressor only, train LLM and projector
                - 'llm': Freeze LLM only, train compressor and projector
                - 'projector': Freeze projector only, train compressor and LLM
                - 'none' or any other value: Train everything (no freezing)
        """
        if mode == 'compress+llm' or mode == 'both':
            # Freeze both compressor and LLM, train only projector
            logger.info("Freezing compressor and LLM. Training only projector.")
            for param in self.compressor.parameters():
                param.requires_grad = False
            for param in self.lm.parameters():
                param.requires_grad = False
                
        elif mode == 'compress':
            # Freeze compressor, train LLM and projector
            logger.info("Freezing compressor. Training LLM and projector.")
            for param in self.compressor.parameters():
                param.requires_grad = False
                
        elif mode == 'llm':
            # Freeze LLM, train compressor and projector
            logger.info("Freezing LLM. Training compressor and projector.")
            for param in self.lm.parameters():
                param.requires_grad = False
                
        elif mode == 'projector':
            # Freeze projector, train compressor and LLM
            raise ValueError("Projector cannot be frozen. Please use 'compress+llm' or 'both' mode.")
                
        elif mode == 'none':
            # Train everything
            logger.info("Training all components (no freezing).")
        else:
            logger.warning(f"Unknown mode '{mode}'. Training all components by default.")

        # If LoRA adapters exist inside modules that were just frozen, re-enable them for training (common for SFT).
        if unfreeze_lora:
            self._unfreeze_lora_adapters(getattr(self, "compressor", None), module_name="compressor")
            self._unfreeze_lora_adapters(getattr(self, "lm", None), module_name="lm")
        
        # Log trainable parameters count
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    def move_padding(
        self,
        inputs: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        padding_side: str = "left"
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return move_padding_to(inputs, attention_mask, labels, padding_side)

    def _resolve_attention_compute_dtype(self, fallback_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        """
        Resolve the dtype used by attention kernels under autocast/mixed precision.
        """
        target_dtype = fallback_dtype
        if torch.is_autocast_enabled():
            if device.type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
                target_dtype = torch.get_autocast_gpu_dtype()
            elif device.type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
                target_dtype = torch.get_autocast_cpu_dtype()
        return target_dtype

    def _cast_past_key_values_dtype(self, past_key_values, target_dtype: torch.dtype):
        """
        Ensure KV cache dtype matches decoder attention compute dtype for FlashAttention.
        """
        if past_key_values is None:
            return None

        if isinstance(past_key_values, DynamicCache):
            casted_cache = DynamicCache()
            # Prefer the current HF cache API (`layers`) when available.
            if hasattr(past_key_values, "layers"):
                for layer_idx, layer in enumerate(past_key_values.layers):
                    layer_key = layer.keys
                    layer_value = layer.values
                    if (layer_key is None) or (layer_value is None):
                        continue
                    casted_cache.update(
                        layer_key.to(dtype=target_dtype),
                        layer_value.to(dtype=target_dtype),
                        layer_idx,
                    )
                return casted_cache

            # Fallback for iterator-style DynamicCache.
            for layer_idx, (layer_key, layer_value) in enumerate(past_key_values):
                if (layer_key is None) or (layer_value is None):
                    continue
                casted_cache.update(
                    layer_key.to(dtype=target_dtype),
                    layer_value.to(dtype=target_dtype),
                    layer_idx,
                )
            return casted_cache

        # Fallback for tuple/list style caches.
        return tuple(
            (layer_key.to(dtype=target_dtype), layer_value.to(dtype=target_dtype))
            for (layer_key, layer_value) in past_key_values
        )

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the compressor model. Must be implemented by subclasses.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for the input
            
        Returns:
            Compressed representation
        """
        pass



class HierarchicalCompressor(EncoderDecoderCompressorBase):
    config_class = HierarchicalCompressorConfig

    """Hierarchical Compressor
    
    Method to overwrite: 
        - compress: adjust the compression logic to fit the hierarchical compressor.
    """

    def __init__(
        self,
        config: HierarchicalCompressorConfig,
        from_pretrained: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(config, from_pretrained, tokenizer=tokenizer)
        self.pooling_method = config.pooling_method
        # Build pooling layer
        pooling_class = get_pooling_factory(self.pooling_method)
        self.pooling_layer = pooling_class(self.config)
        self.use_map_reduce = False

    def enable_map_reduce(self):
        """
        Enable map-reduce compression.
        """
        if self.training:
            logger.warning(
                "Enabling map-reduce during training. This allows long-context training by processing "
                "context in fixed-size segments, but may be slightly slower than a single-pass encoding."
            )
        self.use_map_reduce = True
        logger.info("Map-reduce compression enabled.")

    def disable_map_reduce(self):
        """
        Disable map-reduce compression.
        """
        self.use_map_reduce = False
        logger.info("Map-reduce compression disabled.")

    def _get_token_embeddings(self, input_ids):
        input_ids = input_ids.to(self.lm.device)
        embedding_layer = self.lm.get_input_embeddings()
        embeddings = embedding_layer(input_ids) # same device as the model
        # Cast to model's dtype to ensure consistency
        return embeddings.to(dtype=embedding_layer.weight.dtype)
    

    def _encode_and_prepare_lm_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        ctx_after_query: bool = False,
        add_final_bos: bool = True,
        move_padding_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare the inputs for the base llm model. 
            - encode context into gist tokens
            - modify the attention mask to include the gist tokens
            - add [bos] token to the beginning of the input embeddings
        
        Args:
            input_ids: The input ids of the query.
            attention_mask: The attention mask of the query.
            context_mask: a 2D tensor of bool values mask for the context ids.
                - We assume the number of context ids is the same in each sentence.
                - Currently we assume the context ids are of the same length in each sentence, including the padding tokens. 
                - Please do padding before passing the context mask to the model and reflect the padding in the attention mask.
            
        Return the input embeddings, attention mask, and the gist embeddings length
        """

        device = input_ids.device
        # Get the model's dtype from the embedding layer
        model_dtype = self.lm.get_input_embeddings().weight.dtype

        # Properly index along sequence dimension to preserve batch dimension
        ## Assume that the number of context ids is the same in each sentence.
        context_id_length = context_mask[0].sum()
        query_id_length = context_mask.shape[1] - context_id_length
        
        batch_size = input_ids.shape[0]
        query_ids = input_ids[~context_mask].view(batch_size, query_id_length)
        query_attention_mask = attention_mask[~context_mask].view(batch_size, query_id_length)
        context_ids = input_ids[context_mask].view(batch_size, context_id_length)
        context_attention_mask = attention_mask[context_mask].view(batch_size, context_id_length)

        # Move padding to the front for stable compression
        ## This theoretically won't affect the position of the labels.
        context_ids, context_attention_mask, _ = self.move_padding(
            context_ids, context_attention_mask, padding_side="left"
        )

        if context_ids.shape[1] == 0:
            # Handle empty context case
            gist_embeddings = torch.zeros(
                batch_size, 0, self.config.hidden_size, 
                device=device, dtype=model_dtype
            )
            attention_mask_gists = torch.zeros(
                batch_size, 0, device=device, dtype=torch.long
            )
        else:
            # We allow [bos] token in the compressor inputs
            # Compressor output will be in its loaded dtype (e.g., bf16 if frozen)
            # You can override the compress method to customize the compression logic.
            gist_embeddings, attention_mask_gists = self.compress(context_ids=context_ids, attention_mask=context_attention_mask) 
        
        # Projector: Trainer handles precision via autocast in bf16/fp16 mode
        # Input/output dtype will match the autocast context if enabled
        gist_embeddings = self.projector(gist_embeddings) # Project to embedding space

        # Ensure gist embeddings match the model dtype for consistency
        gist_embeddings = gist_embeddings.to(dtype=model_dtype)
        attention_mask_gists = attention_mask_gists.to(device)

        # Handle query embeddings
        query_embeddings = self._get_token_embeddings(query_ids)
        bos_id = self.tokenizer.bos_token_id
        if (
            bos_id is not None
            and query_ids.shape[1] > 0
            and torch.all(query_ids[:, 0] == bos_id)
        ):
            # remove the [bos] token
            query_embeddings = query_embeddings[:, 1:, :]
            query_attention_mask = query_attention_mask[:, 1:]

        ## Move padding to the front (for generation especially) -> [PAD, GIST, QUESTION]
        if not ctx_after_query:
            # [GIST, QUESTION] (Query might have padding)
            # gist_embeddings: [B, G_Len, H]
            # query_embeddings: [B, Q_Len, H]
            
            # Combine embeddings and masks
            combined_embeddings = torch.cat([gist_embeddings, query_embeddings], dim=1)
            combined_mask = torch.cat([attention_mask_gists, query_attention_mask], dim=1)
        else:
             # [QUESTION, GIST]
            combined_embeddings = torch.cat([query_embeddings, gist_embeddings], dim=1)
            combined_mask = torch.cat([query_attention_mask, attention_mask_gists], dim=1)
            
        # Move padding to the front
        # IMPORTANT: for training (either sft or ntp), we do NOT move padding to the front,
        # because labels only align with the generated tokens which cannot be moved together.
        if move_padding_output:
            input_embeddings, attention_mask_total, _ = self.move_padding(
                combined_embeddings, combined_mask, padding_side="left"
            )
        else:
            input_embeddings = combined_embeddings
            attention_mask_total = combined_mask

        # Add [bos] token
        bos_added = False
        if add_final_bos and bos_id is not None:
            # llama3 is trained with the bos token but qwen3 does not.
            bos_ids = torch.tensor([[bos_id]], dtype=torch.long).repeat(input_embeddings.shape[0], 1).to(device) # [batch, 1]
            bos_embedding = self._get_token_embeddings(bos_ids)
            input_embeddings = torch.cat([bos_embedding, input_embeddings], dim=1)
            attention_mask_total = torch.cat([torch.ones(bos_embedding.shape[0], 1).to(device), attention_mask_total], dim=1)
            bos_added = True

        # Calculate the NTP states starting index. Generation should start from the last gist token.
        if not ctx_after_query:
            # [bos] token + gist tokens length - 1 (minus 1 because we start from 0)
            if bos_added or not add_final_bos:
                 ntp_states_start_index = gist_embeddings.shape[1]
            else:
                 ntp_states_start_index = max(0, gist_embeddings.shape[1] - 1)
        else:
            # no ntp states are needed thus start from the beginning
            ntp_states_start_index = 0

        return input_embeddings, attention_mask_total, ntp_states_start_index


    def compress(self, context_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accept context ids and output the gist embeddings and their attention mask.

        This method is designed to be overridden by subclasses to customize the compression logic.

        Returns:
            gist_embeddings: The gist embeddings.
            attention_mask_gists: The attention mask for the gist embeddings.
        """
        outputs = self.pooling_layer(context_ids, attention_mask, backbone=self.compressor, tokenizer=self.tokenizer)
        gist_embeddings, attention_mask_gists = outputs.pooled, outputs.pooled_mask
        return gist_embeddings, attention_mask_gists

    def map_compression(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        context_mask: Optional[torch.Tensor] = None, 
        segment_length: int = None,
        **kwargs
        ) -> torch.Tensor:
        """
        Compress the context ids and attention mask using the map-reduce compression logic. This is the map step.

        Map reduce will take effect when the input_ids is longer than the segment length.
        In training, map-reduce won't affect the results but can be a bit slower because it requires encoding more times. 
            You can disable map-reduce in training.
        """
        if not self.use_map_reduce:
            return input_ids, attention_mask, context_mask
        
        if (context_mask is None) or (context_mask.sum() == 0):
            # no need to map-reduce because no compression is required
            return input_ids, attention_mask, context_mask
                    
        if segment_length is None:
            if hasattr(self.config, "map_reduce_seg_len"):
                segment_length = self.config.map_reduce_seg_len
            elif hasattr(self.config, "context_length") and (not self.training):
                # we only use the context_length arg for inference 
                # because in training, this arg controls the preprocessing of the context length as well.
                segment_length = self.config.context_length
            else:
                raise ValueError("segment_length is not provided and the context_length is not set in the config. Please set one of them.")

        # Automatically handles the situation where the context length is smaller than the segment length.
        for i in range(0, input_ids.shape[1], segment_length):
            seg_context_ids = input_ids[:, i:i+segment_length]
            seg_attention_mask = attention_mask[:, i:i+segment_length]
            seg_context_mask = context_mask[:, i:i+segment_length]
            yield seg_context_ids, seg_attention_mask, seg_context_mask

    def reduce_compression(
        self, 
        input_embeddings_all: List[torch.Tensor], 
        attention_mask_all: List[torch.Tensor], 
        ntp_states_start_index_all: List[int], 
        **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        if not self.use_map_reduce:
            return input_embeddings_all, attention_mask_all, ntp_states_start_index_all

        device = input_embeddings_all[0].device

        # Concatenate results from each segment
        input_embeddings = torch.cat(input_embeddings_all, dim=1)
        attention_mask_total = torch.cat(attention_mask_all, dim=1)

        # Move padding to the left globally after concatenation to avoid interleaved padding.
        # IMPORTANT: skip during training because labels are right-aligned with the query tokens;
        # moving padding would shift query positions and break label alignment.
        if not self.training:
            input_embeddings, attention_mask_total, _ = self.move_padding(
                input_embeddings, attention_mask_total, padding_side="left"
            )

        # Add the final [bos] token to the input embeddings and attention mask.
        ntp_states_start_index = sum(ntp_states_start_index_all)
        
        if self.tokenizer.bos_token_id is not None:
            bos_ids = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long).repeat(input_embeddings_all[0].shape[0], 1).to(device) # [batch, 1]
            bos_embedding = self._get_token_embeddings(bos_ids)
            input_embeddings = torch.cat([bos_embedding, input_embeddings], dim=1)
            attention_mask_total = torch.cat([torch.ones(bos_embedding.shape[0], 1).to(device), attention_mask_total], dim=1)
        else:
            ntp_states_start_index = max(0, ntp_states_start_index - 1)

        return input_embeddings, attention_mask_total, ntp_states_start_index


    def prepare_lm_inputs(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any
        ) -> torch.Tensor:
        '''
        Encode the context and query, return the representations of the gist tokens and query tokens.

        Assume right padding, do not use left padding (will be moved to the front for generation later automatically).

        Map-reduce:
            Map: Split the context into segments and use _encode_and_prepare_lm_inputs() to encode each segment. 
                - If the input is all context, it will return gist tokens only
                - If the input is all query, it will return query tokens only
                - If the input is a mix of context and query, it will return both gist and query tokens
            No [bos] added in the map step. Will be added in the reduce step.

        
        Args:
            input_ids: The input ids of the query.
            attention_mask: The attention mask of the query.
            context_mask: a 1D tensor of bool values mask for the context ids.
                - Currently we assume the context ids are of the same length in each sentence. 
                - Please do padding before passing the context mask to the model and reflect the padding in the attention mask
            labels: Optional labels for computing language modeling loss.
            **kwargs: Additional arguments for the base model.

        Return:
            outputs: The outputs of the base model.
                - extra attribute: ntp_states - The NTP states if return_ntp_states_only is True, otherwise None.
        '''

        # Handle case where raw input_ids/attention_mask are provided (shouldn't happen with our data collator)
        if context_mask is None:
            return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # check if the context is after the query
        # context_mask is True for context tokens.
        # If the first token is context (True), then context is at the beginning (Context-First).
        # If the first token is NOT context (False), then context is later (Query-First), assuming contiguous segments.
        ctx_after_query = not context_mask[0][0].item()

        if self.use_map_reduce:  
            # map-reduce compression   
            input_embeddings_all = []
            attention_mask_all = []
            context_mask_all = []
            ntp_states_start_index_all = []

            for seg_context_ids, seg_attention_mask, seg_context_mask in self.map_compression(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
            ):
                # Prepare encoded inputs for the base model
                seg_input_embeddings, seg_attention_mask_total, seg_ntp_states_start_index = self._encode_and_prepare_lm_inputs(
                    input_ids=seg_context_ids,
                    attention_mask=seg_attention_mask,
                    context_mask=seg_context_mask,
                    ctx_after_query=ctx_after_query,
                    add_final_bos=False, # no [bos] token in the map step
                    move_padding_output=False # do not move padding in the map step to avoid interleaved padding. Will be moved in the reduce step.
                )

                input_embeddings_all.append(seg_input_embeddings)
                attention_mask_all.append(seg_attention_mask_total)
                context_mask_all.append(seg_context_mask)
                ntp_states_start_index_all.append(seg_ntp_states_start_index)
            
            # Update the input embeddings, attention mask, and ntp states start index with reduce compression.
            input_embeddings, attention_mask_total, ntp_states_start_index = self.reduce_compression(
                input_embeddings_all=input_embeddings_all,
                attention_mask_all=attention_mask_all,
                ntp_states_start_index_all=ntp_states_start_index_all,
            )
        else:
            if self.training:
                # IMPORTANT: for training (either sft or ntp), we should NOT move padding to the front.
                move_padding_output = False
            else:
                move_padding_output = True
            # No map-reduce compression. labels are updated after moving paddings.
            input_embeddings, attention_mask_total, ntp_states_start_index = self._encode_and_prepare_lm_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
                labels=labels,
                ctx_after_query=ctx_after_query,
                add_final_bos=True,
                move_padding_output=move_padding_output
            )
            
        return input_embeddings, attention_mask_total, ntp_states_start_index


    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        context_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        **kwargs: Any
        ) -> torch.Tensor:
        '''
        Forward pass for the compressor.
        Return:
            outputs: The outputs of the base model.
                - extra attribute: ntp_states - The NTP states if return_ntp_states_only is True, otherwise None.
        '''

        ctx_after_query = not context_mask[0][0].item()
        device = input_ids.device

        input_embeddings, attention_mask_total, ntp_states_start_index = self.prepare_lm_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            labels=labels,
            **kwargs
        )
        
        outputs = self.lm(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask_total,
            labels=None,
            **kwargs
        )

        if labels is not None:
            labels = labels.to(device)
            # Prepare labels for loss computation if provided
            if ctx_after_query:
                raise ValueError("When labels are provided to calculate loss, context must be before the query.")
            # Remove the last token because it has no label to predict. Use the generation logits to calculate loss.
            logits = outputs.logits[:, ntp_states_start_index:-1, :].contiguous()
            # Flatten the batch size and calculate CE loss
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return CausalLMOutputWithPast(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return outputs


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs: Any
        ) -> torch.Tensor:

        if context_mask is None:
            # Move padding to the front for generation
            input_ids, attention_mask, _ = self.move_padding(
                input_ids, attention_mask, padding_side="left"
            )
            return self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # check if the context is after the query
        input_embeddings, attention_mask_total, _ = self.prepare_lm_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
            labels=None,
            **kwargs
        )

        outputs = self.lm.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask_total,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        return outputs


class ICAE(HierarchicalCompressor):
    config_class = ICAEConfig
    """Reproduce ICAE"""

    def __init__(
        self,
        config: ICAEConfig,
        from_pretrained: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(config, from_pretrained, tokenizer=tokenizer)
        
        self.num_memory_tokens = config.num_memory_tokens
        
        # Initialize memory embeddings and AE embedding
        # We initialize with randn. 
        self.memory_embeddings = nn.Parameter(torch.randn(1, self.num_memory_tokens, config.hidden_size))
        self.ae_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    @overrides(check_signature=False)
    def compress(self, context_ids: torch.Tensor, attention_mask: torch.Tensor, use_ae:bool=False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # context_ids: [Batch, Seq_Len]
        
        # 1. Get embeddings
        inputs_embeds = self.compressor.get_input_embeddings()(context_ids)
        
        # 2. Append memory embeddings
        batch_size = inputs_embeds.shape[0]
        # Match device and dtype of inputs_embeds
        memory_embeds = self.memory_embeddings.repeat(batch_size, 1, 1).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        
        # Concat: [Text, Memory] (ICAEL3 order: text then memory)
        encoder_inputs = torch.cat([inputs_embeds, memory_embeds], dim=1)
        
        # 3. Adjust attention mask
        # Extend attention mask for memory tokens (attend to everything)
        mem_mask = torch.ones(batch_size, self.num_memory_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
        encoder_mask = torch.cat([attention_mask, mem_mask], dim=1)
        
        # 4. Forward through compressor
        # We use the compressor (which might have LoRA)
        outputs = self.compressor(
            inputs_embeds=encoder_inputs, 
            attention_mask=encoder_mask, 
            output_hidden_states=True,
            output_attentions=True, # for probing
            )
        
        # 5. Extract last num_mem tokens
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
        else:
            # Fallback if output format varies
            hidden_states = outputs[0]

        # [Batch, Seq_Len + Num_Mem, Hidden] -> Take last Num_Mem
        compressed_memory = hidden_states[:, -self.num_memory_tokens:, :]
        
        if use_ae:
            # 6. Append AE embedding (for the decoder to use as prompt/separator)
            ae_embed = self.ae_embedding.repeat(batch_size, 1, 1).to(device=compressed_memory.device, dtype=compressed_memory.dtype)
            
            # Result: [Batch, Num_Mem + 1, Hidden]
            gist_embeddings = torch.cat([compressed_memory, ae_embed], dim=1)
        else:
            gist_embeddings = compressed_memory
        
        gist_mask = torch.ones(gist_embeddings.shape[0], gist_embeddings.shape[1], device=gist_embeddings.device, dtype=torch.long)
        return gist_embeddings, gist_mask

    def stop_gradient(self, mode: str = 'llm', unfreeze_lora: bool = True):
        # unfreeze lora and memory embeddings by default for ICAE
        super().stop_gradient(mode, unfreeze_lora=unfreeze_lora)
        logger.info(f'Unfreeze Lora is set to {unfreeze_lora} for the ICAE model.')
        self.memory_embeddings.requires_grad = True


class ICAEFlex(HierarchicalCompressor):
    config_class = ICAEFlexConfig
    """
    Reproduce ICAE but in a flexible way: 
        - allow the gist tokens to be different lengths in each sentence
        - the gist tokens are added by the tokenizer, and the gist token id is stored in the config.

    Problem: All the gist tokens share the same embedding, while in ICAE, each gist token has a unique embedding.
    Advantage: We can have dymanic compression ratio by adjusting the number of gist tokens.
    """

    def __init__(
        self,
        config: ICAEFlexConfig,
        from_pretrained: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(config, from_pretrained, tokenizer=tokenizer)

        # Add the gist token as the vocab
        self.tokenizer.add_special_tokens({"additional_special_tokens": [config.gist_token]})
        self.vocab_size = len(self.tokenizer)
        self.gist_token_id = self.tokenizer.convert_tokens_to_ids(config.gist_token)
        # Resize embeddings immediately so that the architecture matches the
        # tokenizer size before any checkpoint weights are loaded.
        if not from_pretrained:
            self.resize_token_embeddings(self.vocab_size, mean_resizing=False)


    def _get_token_embeddings(self, input_ids):
        input_ids = input_ids.to(self.lm.device)
        embedding_layer = self.lm.get_input_embeddings()
        embeddings = embedding_layer(input_ids) # same device as the model
        # Cast to model's dtype to ensure consistency
        return embeddings.to(dtype=embedding_layer.weight.dtype)

    def stop_gradient(self, mode: str = 'llm', unfreeze_lora: bool = True):
        # unfreeze lora and memory embeddings by default for ICAEFlex
        super().stop_gradient(mode, unfreeze_lora=unfreeze_lora)
        self.memory_embeddings.requires_grad = True # always keep memory embeddings trainable

    @overrides(check_signature=False)
    def compress(self, context_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # The compressor will output the states of the context tokens and the gist tokens in ICAE
        # Therefore, we need to select the gist tokens from the output.
        gist_id_mask = context_ids == self.gist_token_id
        # WARNING: This assumes all sequences in the batch have the same number of gist tokens.
        gist_id_length = gist_id_mask[0].sum()
        
        outputs = self.compressor(input_ids=context_ids, attention_mask=attention_mask)
        
        # Extract hidden states from the output object
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
            
        # Select gist tokens: [Total_Gist_Tokens, Hidden]
        selected_gists = hidden_states[gist_id_mask]
        
        # Reshape to [Batch, Gist_Len, Hidden]
        embeddings = selected_gists.view(-1, gist_id_length, hidden_states.shape[-1])
        gist_mask = torch.ones(embeddings.shape[0], embeddings.shape[1], device=embeddings.device, dtype=torch.long)
        return embeddings, gist_mask

class FiveHundredX(EncoderDecoderCompressorBase):
    config_class = FiveHundredXConfig
    
    def __init__(
        self,
        config: FiveHundredXConfig,
        from_pretrained: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(config, from_pretrained, tokenizer=tokenizer)
        self.num_memory_tokens = config.num_memory_tokens
        self.use_map_reduce = False
        
        if not from_pretrained:
            # Initialize memory embeddings
            # We need to deduce dtype and device from the model
            if hasattr(self.lm, 'device'):
                device = self.lm.device
            else:
                device = torch.device('cpu')
                
            # Get dtype from model parameters
            try:
                dtype = next(self.lm.parameters()).dtype
            except:
                dtype = torch.float32

            self.memory_embeddings = nn.Parameter(
                torch.randn(1, self.num_memory_tokens, config.hidden_size, dtype=dtype, device=device)
                )

    @classmethod
    def from_pretrained_submodules(
        cls,
        config: FiveHundredXConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        # Call the base class method to load models
        model = super().from_pretrained_submodules(config, tokenizer=tokenizer, **kwargs)
        
        # Initialize memory embeddings on the correct device/dtype
        device = model.lm.device
        dtype = next(model.lm.parameters()).dtype
        model.memory_embeddings = nn.Parameter(
            torch.randn(1, config.num_memory_tokens, config.hidden_size, dtype=dtype, device=device)
            )
        return model

    def stop_gradient(self, mode: str = 'llm', unfreeze_lora: bool = True):
        # For FiveHundredX, 'llm' mode means freeze LLM base, train LoRA (compressor) and Memory.
        super().stop_gradient(mode, unfreeze_lora=unfreeze_lora)
        logger.info(f'Unfreeze Lora is set to {unfreeze_lora} for the 500x model.')
        self.memory_embeddings.requires_grad = True

    def enable_map_reduce(self):
        """
        Enable map-reduce compression.
        """
        if self.training:
            logger.warning(
                "Enabling map-reduce during training. This allows long-context training by processing "
                "context in fixed-size segments, but may be slightly slower than a single-pass encoding."
            )
        self.use_map_reduce = True
        logger.info("Map-reduce compression enabled.")

    def disable_map_reduce(self):
        """
        Disable map-reduce compression.
        """
        self.use_map_reduce = False
        logger.info("Map-reduce compression disabled.")

    def _compress_segment(self, text_tokens, context_attention_mask, move_padding=True, position_offset=0):
        device = text_tokens.device
        batch_size = text_tokens.shape[0]
        
        if move_padding:
            text_tokens, context_attention_mask, _ = self.move_padding(
                 text_tokens, context_attention_mask, padding_side="left"
            )

        # Use compressor which has LoRA
        text_tok_embeddings = self.compressor.get_input_embeddings()(text_tokens).to(device)
        memory_tok_embeddings = self.memory_embeddings.repeat(batch_size, 1, 1).to(device)
        
        # Concatenate: [Text, Memory]
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        
        # Create extended attention mask for [Text, Memory]
        # Text part uses provided mask. Memory part is fully attended.
        memory_mask = torch.ones(batch_size, self.num_memory_tokens, device=device, dtype=context_attention_mask.dtype)
        encoder_attention_mask = torch.cat([context_attention_mask, memory_mask], dim=1)

        # Create position_ids for the segment
        seq_len = encoder_input_embeddings.shape[1]
        position_ids = torch.arange(position_offset, position_offset + seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Forward pass through LoRA-enabled compressor
        encoder_output = self.compressor(
            inputs_embeds=encoder_input_embeddings, 
            attention_mask=encoder_attention_mask,
            position_ids=position_ids, 
            use_cache=True,
        )
        past_key_values = encoder_output.past_key_values
        
        # Extract KV cache for memory tokens (last num_memory_tokens)
        trimmed_past_key_values = DynamicCache()
        for layer_idx, (layer_key, layer_value) in enumerate(past_key_values):
            trimmed_past_key_values.update(
                layer_key[:, :, -self.num_memory_tokens:, :], 
                layer_value[:, :, -self.num_memory_tokens:, :], 
                layer_idx
            )
        return trimmed_past_key_values

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any
        ) -> torch.Tensor:
        """Rewrite the forward method to support kv-cache passing from encoder to decoder"""
        
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Identify context and query
        if context_mask is None:
             # Assume all input is text to be compressed/reconstructed (Autoencoder mode)
             text_tokens = input_ids
             query_tokens = None
             target_tokens = labels if labels is not None else input_ids
             # Create a simple mask for context
             context_attention_mask = attention_mask
             query_attention_mask = None
        else:             
             # Split based on mask
             # Assume contiguous context then query
             context_id_length = context_mask[0].sum()
             query_id_length = context_mask.shape[1] - context_id_length
             
             text_tokens = input_ids[context_mask].view(batch_size, context_id_length)
             context_attention_mask = attention_mask[context_mask].view(batch_size, context_id_length)
             
             query_tokens = input_ids[~context_mask].view(batch_size, query_id_length)
             query_attention_mask = attention_mask[~context_mask].view(batch_size, query_id_length)
             
             if labels is not None:
                 target_tokens = labels
             else:
                 target_tokens = query_tokens


        # --- Compressing --
        trimmed_past_key_values = None

        if self.use_map_reduce and context_attention_mask is not None and context_mask is not None:
             # Map Reduce Compression
             if hasattr(self.config, "context_length"):
                segment_length = self.config.context_length
             else:
                segment_length = 256 # Default fallback

             kv_caches_list = []
             current_position_offset = 0
             
             for i in range(0, text_tokens.shape[1], segment_length):
                 seg_text = text_tokens[:, i:i+segment_length]
                 seg_mask = context_attention_mask[:, i:i+segment_length]
                 
                 # Compress segment
                 seg_kv = self._compress_segment(
                     seg_text, 
                     seg_mask, 
                     move_padding=True,
                     position_offset=current_position_offset
                 )
                 kv_caches_list.append(seg_kv)
                 current_position_offset += seg_text.shape[1]
            
             # Concatenate KV caches
             trimmed_past_key_values = DynamicCache()
             num_layers = len(kv_caches_list[0])
             for layer_idx in range(num_layers):
                 keys = torch.cat([kv[layer_idx][0] for kv in kv_caches_list], dim=2)
                 values = torch.cat([kv[layer_idx][1] for kv in kv_caches_list], dim=2)
                 trimmed_past_key_values.update(keys, values, layer_idx)
                 
             if context_mask is None and target_tokens is not None:
                 # In Map-Reduce Autoencoder (rare), we need to handle target_tokens if we didn't move global padding.
                 # For simplicity, we assume Map-Reduce is mostly for Context-Query.
                 pass

        else:
            # Standard Compression
            # Move padding tokens to the front for text_tokens and context_attention_mask
            if context_attention_mask is not None:
                text_tokens, context_attention_mask, target_tokens_if_autoencoder = self.move_padding(
                    text_tokens, 
                    context_attention_mask, 
                    labels=target_tokens if context_mask is None else None,
                    padding_side="left"
                )
                if context_mask is None and target_tokens is not None:
                    # in autoencoder mode, we also move the labels to the front.
                    target_tokens = target_tokens_if_autoencoder
            
            # Use helper
            trimmed_past_key_values = self._compress_segment(text_tokens, context_attention_mask, move_padding=False) # Padding already moved

                 
        # Move padding tokens to the front for query_tokens if they exist
        # IMPORTANT: for training (either sft or ntp), we do NOT move padding to the front for query,
        # because labels only align with the generated tokens which cannot be moved together if they are right-padded.
        # For inference, we move padding to the front for query.
        if (not self.training) and (query_tokens is not None) and (query_attention_mask is not None):
             # If target_tokens corresponds to query, we move it too
             query_tokens, query_attention_mask, target_tokens = self.move_padding(
                 query_tokens, query_attention_mask, labels=target_tokens, padding_side="left"
             )

        # --- Decoding --
        if query_tokens is None:
             # Reconstruct input (Autoencoder)
             query_input_ids = text_tokens
        else:
             query_input_ids = query_tokens

        # Add BOS token to query input
        if self.tokenizer.bos_token_id is not None:
            bos_tokens = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size, device=device)
            bos_embeddings = self.lm.get_input_embeddings()(bos_tokens)
            query_embeddings = self.lm.get_input_embeddings()(query_input_ids)
            
            decoder_input_embeddings = torch.cat((bos_embeddings, query_embeddings), dim=1)
        else:
            decoder_input_embeddings = self.lm.get_input_embeddings()(query_input_ids)

        target_attn_dtype = self._resolve_attention_compute_dtype(
            fallback_dtype=decoder_input_embeddings.dtype,
            device=decoder_input_embeddings.device,
        )
        trimmed_past_key_values = self._cast_past_key_values_dtype(trimmed_past_key_values, target_attn_dtype)

        # Forward pass through base model (self.lm does NOT have LoRA)
        # Use trimmed_past_key_values as prefix
        # FIX: Calculate start position for decoder to match the positions of the memory tokens
        memory_length = trimmed_past_key_values.get_seq_length() if isinstance(trimmed_past_key_values, DynamicCache) else trimmed_past_key_values[0][0].shape[2]
        start_pos = text_tokens.shape[1] + memory_length
        decoder_seq_len = decoder_input_embeddings.shape[1]
        decoder_position_ids = torch.arange(start_pos, start_pos + decoder_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        decoder_output = self.lm(
            inputs_embeds=decoder_input_embeddings, 
            past_key_values=trimmed_past_key_values,
            position_ids=decoder_position_ids
        )
            
        logits = decoder_output.logits
        
        loss = None
        if target_tokens is not None:
            if self.tokenizer.bos_token_id is None:
                # No [bos] case (qwen): inputs = query_tokens, so predict token[t+1] from state[t]
                logits_for_loss = logits[:, :-1, :].contiguous()
                labels_for_loss = target_tokens[:, 1:].contiguous()
            else:
                # With [bos] case (llama): inputs = [BOS] + query_tokens, so predict query_tokens[t] from state[t]
                logits_for_loss = logits[:, :-1, :].contiguous()
                labels_for_loss = target_tokens.contiguous()

            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits_for_loss.view(-1, logits_for_loss.size(-1)),
                labels_for_loss.view(-1),
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_output.past_key_values, 
            hidden_states=decoder_output.hidden_states,
            attentions=decoder_output.attentions,
        )


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generate text using the FiveHundredX model.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            context_mask: Mask identifying context tokens (to be compressed).
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional arguments for generation.
            
        Returns:
            Generated token IDs.
        """
        
        # If no context mask is provided, we assume standard generation using the base model
        if context_mask is None:
             # Move padding to the front for generation
            input_ids, attention_mask, _ = self.move_padding(
                input_ids, attention_mask, padding_side="left"
            )
            return self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 1. Split inputs into Context and Query
        context_id_length = context_mask[0].sum()
        query_id_length = context_mask.shape[1] - context_id_length
        
        text_tokens = input_ids[context_mask].view(batch_size, context_id_length)
        context_attention_mask = attention_mask[context_mask].view(batch_size, context_id_length)
        
        query_tokens = input_ids[~context_mask].view(batch_size, query_id_length)
        query_attention_mask = attention_mask[~context_mask].view(batch_size, query_id_length)

        # 2. Compress Context
        trimmed_past_key_values = None
        
        if self.use_map_reduce:
             if hasattr(self.config, "context_length"):
                segment_length = self.config.context_length
             else:
                segment_length = 256
                
             kv_caches_list = []
             current_position_offset = 0
             
             for i in range(0, text_tokens.shape[1], segment_length):
                 seg_text = text_tokens[:, i:i+segment_length]
                 seg_mask = context_attention_mask[:, i:i+segment_length]
                 
                 seg_kv = self._compress_segment(
                     seg_text, 
                     seg_mask, 
                     move_padding=True,
                     position_offset=current_position_offset
                 )
                 kv_caches_list.append(seg_kv)
                 current_position_offset += seg_text.shape[1]
            
             # Concatenate KV caches
             trimmed_past_key_values = DynamicCache()
             num_layers = len(kv_caches_list[0])
             for layer_idx in range(num_layers):
                 keys = torch.cat([kv[layer_idx][0] for kv in kv_caches_list], dim=2)
                 values = torch.cat([kv[layer_idx][1] for kv in kv_caches_list], dim=2)
                 trimmed_past_key_values.update(keys, values, layer_idx)

        else:
            # Standard Compression
            # Move padding tokens to the front for text_tokens and context_attention_mask
            text_tokens, context_attention_mask, _ = self.move_padding(
                text_tokens, context_attention_mask, 
                padding_side="left"
            )
            trimmed_past_key_values = self._compress_segment(text_tokens, context_attention_mask, move_padding=False)

            # 3. Prepare Query for Generation
            # query_tokens, query_attention_mask, _ = self.move_padding(
            #      query_tokens, query_attention_mask, padding_side="left"
            #  )
            
            # Add BOS token to query input
            if self.tokenizer.bos_token_id is not None:
                bos_tokens = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size, device=device)
                decoder_input_ids = torch.cat((bos_tokens, query_tokens), dim=1)
            else:
                decoder_input_ids = query_tokens

            target_attn_dtype = self._resolve_attention_compute_dtype(
                fallback_dtype=self.lm.get_input_embeddings().weight.dtype,
                device=device,
            )
            trimmed_past_key_values = self._cast_past_key_values_dtype(trimmed_past_key_values, target_attn_dtype)
            
            # Prepare attention mask
            # IMPORTANT (generation correctness): mask out padding in the query.
            #
            # In training `forward()`, the decoder is currently called without an `attention_mask`, which means
            # padding tokens can participate in attention. However, during autoregressive generation, if the
            # right-padded `<pad>` tokens are treated as "real" context (mask=1), `generate()` may start from
            # a `<pad>` position and immediately emit EOS or produce unstable outputs.
            #
            # Therefore, for inference we use the standard padding mask: BOS=1 followed by query_attention_mask.
            #
            # Fallback: if query_attention_mask is missing, infer it from pad_token_id.
            if query_attention_mask is None:
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    raise ValueError("tokenizer.pad_token_id is None but query_attention_mask is required for padded generation.")
                query_attention_mask = (query_tokens != pad_token_id).to(dtype=attention_mask.dtype, device=device)


            attn_mask_offset = 1 if self.tokenizer.bos_token_id is not None else 0
            decoder_attention_mask = torch.cat(
                (
                    torch.ones(batch_size, attn_mask_offset, device=device, dtype=query_attention_mask.dtype),
                    query_attention_mask,
                ),
                dim=1,
            )
            
            # Extend attention mask for past key values
            past_length = trimmed_past_key_values.get_seq_length() if isinstance(trimmed_past_key_values, DynamicCache) else trimmed_past_key_values[0][0].shape[2]
            past_mask = torch.ones(batch_size, past_length, device=device, dtype=decoder_attention_mask.dtype)
            combined_attention_mask = torch.cat([past_mask, decoder_attention_mask], dim=1)
            
            # IMPORTANT (Transformers >=4.4x): if we pass `past_key_values`, HF assumes `input_ids` includes that prefix
            # and will slice cache_position by `past_length`. Here, the past is an *external* prefix (compressed context),
            # so we must provide a cache_position that starts at `past_length` to avoid creating an empty cache_position.
            
            # FIX: Align cache_position with the actual positions of the memory tokens to preserve RoPE relative distances.
            # Memory tokens are located after the context (including padding), so the query should start after them.
            # start_pos = int(context_id_length.item()) + past_length
            # start_pos = text_tokens.shape[1] + past_length 
            # start_pos = int(context_id_length.item())
            # start_pos = past_length
            start_pos = text_tokens.shape[1] + past_length 

            cache_position = torch.arange(
                start_pos,
                start_pos + decoder_input_ids.shape[1],
                device=device,
                dtype=torch.long,
            )
            
            # Here the output will contain the orignal decoder_input_ids which is not as expected for a compression model.
            # Thus, we remove the prefix decoder_input_ids from the output.
            outputs =  self.lm.generate(
                input_ids=decoder_input_ids,
                attention_mask=combined_attention_mask,
                past_key_values=trimmed_past_key_values,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            outputs = outputs[:, decoder_input_ids.shape[1]:]
            return outputs
   


class SAC(EncoderDecoderCompressorBase):
    """SAC (Semantic Anchors for Context Compression) baseline.

    Compresses context by uniformly subsampling token positions, adding learnable role tokens,
    and passing the resulting KV cache (with bidirectional attention) to a frozen decoder.

    Reference: original SAC implementation at src/baselines/SAC/model/modeling.py
    """
    config_class = SACConfig

    def __init__(
        self,
        config: SACConfig,
        from_pretrained: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__(config, from_pretrained, tokenizer=tokenizer)
        self.chunk_size = config.chunk_size
        self.compress_ratio = config.compress_ratio
        self.mem_size = config.mem_size

        if not from_pretrained:
            self._init_sac_parameters()

    def _init_sac_parameters(self):
        """Initialize SAC-specific learnable parameters (role tokens and LM special token)."""
        hidden_size = self.lm.config.hidden_size

        # Role tokens: added to uniformly-subsampled positions in each chunk (SAC line 34, 41)
        self.role_tokens = nn.Parameter(
            torch.zeros(self.mem_size, hidden_size)
        )
        nn.init.normal_(self.role_tokens, mean=0.0, std=0.02)

        # LM special token: prepended to decoder input (SAC line 35, 39-42, index 1)
        self.lm_special_token = nn.Parameter(
            torch.zeros(1, hidden_size)
        )
        embed_weight = self.lm.get_input_embeddings().weight
        if embed_weight.device.type == "meta":
            # During HF from_pretrained with low_cpu_mem_usage, weights are meta tensors.
            # Use sensible defaults; real weights will be loaded from checkpoint afterward.
            nn.init.normal_(self.lm_special_token, mean=0.0, std=0.02)
        else:
            mean = torch.mean(embed_weight).item()
            std = torch.std(embed_weight).item()
            nn.init.normal_(self.lm_special_token, mean=mean, std=std)

    @classmethod
    def from_pretrained_submodules(
        cls,
        config: SACConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        model = super().from_pretrained_submodules(config, tokenizer=tokenizer, **kwargs)
        model._init_sac_parameters()
        return model

    def stop_gradient(self, mode: str = 'llm', unfreeze_lora: bool = True):
        super().stop_gradient(mode, unfreeze_lora=unfreeze_lora)
        # SAC's role tokens and special token are always trainable
        self.role_tokens.requires_grad = True
        self.lm_special_token.requires_grad = True

    @staticmethod
    def _get_uniform_position_ids(x_1: int, x_n: int, ratio: int, device: torch.device) -> torch.Tensor:
        """Uniformly subsample position IDs at `ratio` intervals. Exact copy of SAC line 162-167."""
        start = x_1 + (ratio - 1) // 2
        end = x_n
        if start >= end:
            start = x_1
        return torch.arange(start, end, step=ratio, device=device).unsqueeze(0)

    def _compress_chunks(
        self,
        text_tokens: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> Tuple[DynamicCache, int]:
        """Core compression: chunk context, add role tokens, extract KV cache at subsampled positions.

        Faithful to SAC's compress() method (modeling.py lines 169-249).

        Returns:
            (encoder_past_key_values, end_idx): KV cache and last context position index.
        """
        device = text_tokens.device
        batch_size, total_length = text_tokens.shape
        num_chunks = math.ceil(total_length / self.chunk_size)

        all_trimmed_past_key_values = []
        end_idx = 0

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, total_length)
            chunk_input_ids = text_tokens[:, start_idx:end_idx]

            # Embed tokens (SAC line 192)
            inputs_embeds = self.compressor.get_input_embeddings()(chunk_input_ids)
            seq_len = inputs_embeds.shape[1]
            emb_size = inputs_embeds.shape[2]

            # 1-indexed position IDs (SAC line 196)
            position_ids = torch.arange(
                start_idx + 1, end_idx + 1, device=device
            ).unsqueeze(0)

            # Uniform subsampling of memory positions (SAC line 198)
            mem_position_ids = self._get_uniform_position_ids(
                x_1=start_idx + 1, x_n=end_idx + 1,
                ratio=self.compress_ratio, device=device,
            )
            current_mem_size = mem_position_ids.size(1)

            # Add role tokens at subsampled positions (SAC lines 205-208)
            encode_inputs_embeds = inputs_embeds.clone()
            role_embeds = self.role_tokens[:current_mem_size, :].unsqueeze(0).expand(
                batch_size, current_mem_size, emb_size
            )
            mem_real_idx = mem_position_ids.squeeze(0) - 1 - start_idx  # convert to 0-indexed local
            encode_inputs_embeds.index_add_(1, mem_real_idx, role_embeds)

            # Bidirectional attention: all-zeros 4D mask (SAC line 210)
            attention_mask = torch.zeros(
                1, 1, seq_len, seq_len,
                device=device, dtype=encode_inputs_embeds.dtype,
            )

            # Forward through encoder with KV cache (SAC lines 218-222)
            outputs = self.compressor(
                inputs_embeds=encode_inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

            # Trim KV cache to memory positions only (SAC lines 238-242)
            past_key_values = outputs.past_key_values
            trimmed = DynamicCache()
            for layer_idx, (layer_key, layer_value) in enumerate(past_key_values):
                trimmed.update(
                    layer_key[:, :, mem_real_idx, :],
                    layer_value[:, :, mem_real_idx, :],
                    layer_idx,
                )
            all_trimmed_past_key_values.append(trimmed)

        # Concatenate KV caches across chunks (SAC line 247)
        num_layers = len(all_trimmed_past_key_values[0])
        encoder_past_key_values = DynamicCache()
        for layer_idx in range(num_layers):
            keys = torch.cat(
                [kv[layer_idx][0] for kv in all_trimmed_past_key_values], dim=2
            )
            values = torch.cat(
                [kv[layer_idx][1] for kv in all_trimmed_past_key_values], dim=2
            )
            encoder_past_key_values.update(keys, values, layer_idx)

        return encoder_past_key_values, end_idx

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # --- Split context and query (same pattern as FiveHundredX) ---
        if context_mask is None:
            # No context to compress — fall back to standard LM forward
            return self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        context_id_length = context_mask[0].sum()
        query_id_length = context_mask.shape[1] - context_id_length

        text_tokens = input_ids[context_mask].view(batch_size, context_id_length)
        context_attention_mask = attention_mask[context_mask].view(batch_size, context_id_length)

        query_tokens = input_ids[~context_mask].view(batch_size, query_id_length)

        target_tokens = labels if labels is not None else query_tokens

        # Move padding to front for stable compression
        text_tokens, context_attention_mask, _ = self.move_padding(
            text_tokens, context_attention_mask, padding_side="left"
        )

        # --- Compress context ---
        encoder_past_key_values, end_idx = self._compress_chunks(
            text_tokens, context_attention_mask
        )

        # --- Build decoder input (SAC lines 90-94) ---
        # Embed query tokens, shift right by 1, prepend learned [LM] token
        query_embeddings = self.lm.get_input_embeddings()(query_tokens[:, :-1])
        lm_token = self.lm_special_token.unsqueeze(0).expand(batch_size, 1, -1)
        decoder_input_embeddings = torch.cat([lm_token, query_embeddings], dim=1)

        # Decoder position IDs: continue from context end (SAC line 98)
        decoder_seq_len = decoder_input_embeddings.shape[1]
        decoder_position_ids = torch.arange(
            end_idx, end_idx + decoder_seq_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        # Cast KV cache dtype for mixed precision compatibility
        target_attn_dtype = self._resolve_attention_compute_dtype(
            fallback_dtype=decoder_input_embeddings.dtype,
            device=device,
        )
        encoder_past_key_values = self._cast_past_key_values_dtype(
            encoder_past_key_values, target_attn_dtype
        )

        # --- Decode with compressed context as KV cache (SAC line 102) ---
        decoder_output = self.lm(
            inputs_embeds=decoder_input_embeddings,
            past_key_values=encoder_past_key_values,
            position_ids=decoder_position_ids,
        )

        # --- Loss (SAC lines 106-109) ---
        logits = decoder_output.logits
        loss = None
        if target_tokens is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.contiguous().view(-1, logits.size(-1)),
                target_tokens.contiguous().view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states,
            attentions=decoder_output.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive generation, faithful to SAC's lm_inference (lines 251-284)."""
        if context_mask is None:
            input_ids, attention_mask, _ = self.move_padding(
                input_ids, attention_mask, padding_side="left"
            )
            return self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Split context and query
        context_id_length = context_mask[0].sum()
        query_id_length = context_mask.shape[1] - context_id_length

        text_tokens = input_ids[context_mask].view(batch_size, context_id_length)
        context_attention_mask = attention_mask[context_mask].view(batch_size, context_id_length)

        query_tokens = input_ids[~context_mask].view(batch_size, query_id_length)

        # Move padding and compress
        text_tokens, context_attention_mask, _ = self.move_padding(
            text_tokens, context_attention_mask, padding_side="left"
        )
        encoder_past_key_values, end_idx = self._compress_chunks(
            text_tokens, context_attention_mask
        )

        # Cast KV cache dtype
        target_attn_dtype = self._resolve_attention_compute_dtype(
            fallback_dtype=self.lm.get_input_embeddings().weight.dtype,
            device=device,
        )
        encoder_past_key_values = self._cast_past_key_values_dtype(
            encoder_past_key_values, target_attn_dtype
        )

        # Build initial decoder input: [LM] + query token embeddings (SAC lines 253-257)
        query_embeddings = self.lm.get_input_embeddings()(query_tokens)
        lm_token = self.lm_special_token.unsqueeze(0).expand(batch_size, 1, -1)
        next_inputs_embeds = torch.cat([lm_token, query_embeddings], dim=1)

        # Position IDs: start from end_idx (SAC line 260)
        seq_len = next_inputs_embeds.shape[1]
        next_position_ids = torch.arange(
            end_idx, end_idx + seq_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        # Autoregressive generation loop (SAC lines 267-284)
        past_key_values = encoder_past_key_values
        generated_ids = []

        for _ in range(max_new_tokens):
            outputs = self.lm(
                inputs_embeds=next_inputs_embeds,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logit = outputs.logits[:, -1]  # [B, V]
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(logit, dim=-1)  # [B]
            generated_ids.append(next_token_id)

            # Prepare next step input
            next_inputs_embeds = self.lm.get_input_embeddings()(next_token_id).unsqueeze(1)
            next_position_ids = next_position_ids[:, -1:] + 1

            # Stop on EOS (batch_size=1 assumption, matching SAC original)
            if batch_size == 1 and next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return torch.stack(generated_ids, dim=1)  # [B, generated_len]


__all__ = [
    "EncoderDecoderCompressorBaseConfig",
    "PartialLLMModel",
    "ProjectionMLP",
    "EncoderDecoderCompressorBase",
    "HierarchicalCompressorConfig",
    "ICAEConfig",
    "ICAEFlexConfig",
    "FiveHundredXConfig",
    "SACConfig",
    "HierarchicalCompressor",
    "ICAE",
    "ICAEFlex",
    "FiveHundredX",
    "SAC",
    "get_model_factory_from_config",
    "get_model_factory",
    ]
