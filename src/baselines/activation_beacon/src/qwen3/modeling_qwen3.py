# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3 model with Activation Beacon support.

Adapted from modeling_qwen2.py (beacon version) for Qwen3's architecture:
 - q_norm / k_norm (per-head RMSNorm applied after q/k projections)
 - head_dim as an explicit config parameter
 - attention_bias config param (defaults to False)
 - rope_parameters dict in config (instead of a single rope_theta float)
 - layer_types list per layer (sliding vs full attention) — ignored in beacon mode
"""

import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from .configuration_qwen3 import Qwen3Config

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from ..modeling_beacon import Memory
from ..modeling_utils import optional_grad_ctx, compute_loss, get_rope, ModelOutput


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"
_CONFIG_FOR_DOC = "Qwen3Config"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# ---------------------------------------------------------------------------
# Modules: RMSNorm, MLP, RoPE helpers
# ---------------------------------------------------------------------------

class Qwen3RMSNorm(nn.Module):
    """Qwen3RMSNorm is equivalent to T5LayerNorm."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    (batch, num_key_value_heads, seqlen, head_dim) → (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_rope_theta(config: Qwen3Config) -> float:
    """Extract rope_theta from Qwen3's rope_parameters dict or direct attribute.

    Qwen3 config.json may store the theta as:
      - config.rope_parameters["rope_theta"]  (new HF format)
      - config.rope_theta                     (legacy flat format, passed via **kwargs)
    We try both so that all checkpoint flavors are supported.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is not None:
        if isinstance(rope_params, dict):
            return rope_params.get("rope_theta", 10000.0)
        return getattr(rope_params, "rope_theta", 10000.0)
    # Fall back to a direct attribute (stored via **kwargs in the config JSON)
    return getattr(config, "rope_theta", 10000.0)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Qwen3Attention(nn.Module):
    """Multi-headed attention with Activation Beacon support.

    Differences from Qwen2Attention:
    - q_norm / k_norm: per-head RMSNorm applied after q/k projections (before RoPE)
    - head_dim read from config.head_dim (may differ from hidden_size // num_heads)
    - attention_bias = config.attention_bias (defaults to False for Qwen3)
    """

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        attn_bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attn_bias)

        # Qwen3-specific: per-head query/key normalisation
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_eps)

        rope_theta = _get_rope_theta(config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rope_theta,
            config.max_position_embeddings,
            getattr(config, "rope_scaling", None),
        )

        # ------------------------------------------------------------------ #
        # Beacon extra parameters                                              #
        # ------------------------------------------------------------------ #
        if "q" in config.beacon_param:
            self.beacon_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True
        if "k" in config.beacon_param:
            self.beacon_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
            self.beacon_k_proj.weight.data.zero_()
            self.beacon_k_proj._is_hf_initialized = True
        if "v" in config.beacon_param:
            self.beacon_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
            self.beacon_v_proj.weight.data.zero_()
            self.beacon_v_proj._is_hf_initialized = True
        if "o" in config.beacon_param:
            self.beacon_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attn_bias)
            self.beacon_o_proj.weight.data.zero_()
            self.beacon_o_proj._is_hf_initialized = True

    def _init_beacon_proj(self, missing_keys):
        """Initialize beacon projection weights from the corresponding ordinal weights."""
        beacon_param = self.config.beacon_param

        if is_deepspeed_zero3_enabled():
            import deepspeed

            def _copy_ds(beacon_w, src_w, beacon_b=None, src_b=None):
                params = [beacon_w, src_w]
                if beacon_b is not None:
                    params += [beacon_b, src_b]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (beacon_w.sum(-1) == 0).any() or (beacon_w > 1e29).any():
                        beacon_w.data[:] = src_w.data
                        if beacon_b is not None:
                            beacon_b.data[:] = src_b.data

            if "q" in beacon_param:
                _copy_ds(self.beacon_q_proj.weight, self.q_proj.weight,
                         self.beacon_q_proj.bias if self.q_proj.bias is not None else None,
                         self.q_proj.bias)
            if "k" in beacon_param:
                _copy_ds(self.beacon_k_proj.weight, self.k_proj.weight,
                         self.beacon_k_proj.bias if self.k_proj.bias is not None else None,
                         self.k_proj.bias)
            if "v" in beacon_param:
                _copy_ds(self.beacon_v_proj.weight, self.v_proj.weight,
                         self.beacon_v_proj.bias if self.v_proj.bias is not None else None,
                         self.v_proj.bias)
            if "o" in beacon_param:
                _copy_ds(self.beacon_o_proj.weight, self.o_proj.weight,
                         self.beacon_o_proj.bias if self.o_proj.bias is not None else None,
                         self.o_proj.bias)
        else:
            if "q" in beacon_param and any("beacon_q_proj" in k for k in missing_keys):
                self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                if self.q_proj.bias is not None:
                    self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in beacon_param and any("beacon_k_proj" in k for k in missing_keys):
                self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                if self.k_proj.bias is not None:
                    self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param and any("beacon_v_proj" in k for k in missing_keys):
                self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                if self.v_proj.bias is not None:
                    self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in beacon_param and any("beacon_o_proj" in k for k in missing_keys):
                self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                if self.o_proj.bias is not None:
                    self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data

    # ---------------------------------------------------------------------- #
    # Beacon-aware projection helpers                                         #
    # ---------------------------------------------------------------------- #

    def qkv_proj_with_beacon(self, hidden_states, beacon_size, beacon_indices):
        """Return (query, key, value) raw projections, shape (batch, seq, heads*head_dim).

        When beacon tokens are present the appropriate beacon/ordinal projection
        matrix is selected per-token.  The q_norm / k_norm are applied by the
        caller *after* reshaping to (batch, seq, heads, head_dim).
        """
        if beacon_size > 0:
            cur_beacon_indices = beacon_indices[-hidden_states.shape[1]:]

            if "q" in self.config.beacon_param:
                ordinal_q = self.q_proj(hidden_states)
                beacon_q = self.beacon_q_proj(hidden_states)
                query_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_q, beacon_q)
                if (cur_beacon_indices == 2).any():
                    query_states = query_states.clone()
                    query_states[:, cur_beacon_indices == 2] = beacon_q[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                query_states = self.q_proj(hidden_states)

            if "k" in self.config.beacon_param:
                ordinal_k = self.k_proj(hidden_states)
                beacon_k = self.beacon_k_proj(hidden_states)
                key_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_k, beacon_k)
                if (cur_beacon_indices == 2).any():
                    key_states = key_states.clone()
                    key_states[:, cur_beacon_indices == 2] = beacon_k[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                key_states = self.k_proj(hidden_states)

            if "v" in self.config.beacon_param:
                ordinal_v = self.v_proj(hidden_states)
                beacon_v = self.beacon_v_proj(hidden_states)
                value_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_v, beacon_v)
                if (cur_beacon_indices == 2).any():
                    value_states = value_states.clone()
                    value_states[:, cur_beacon_indices == 2] = beacon_v[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                value_states = self.v_proj(hidden_states)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states

    def o_proj_with_beacon(self, attn_output, beacon_size, beacon_indices):
        if beacon_size > 0:
            cur_beacon_indices = beacon_indices[-attn_output.shape[1]:]
            if "o" in self.config.beacon_param:
                ordinal_out = self.o_proj(attn_output)
                beacon_out = self.beacon_o_proj(attn_output)
                attn_output = torch.where((cur_beacon_indices == 0)[:, None], ordinal_out, beacon_out)
            else:
                attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = q_len
        past_key, past_value, beacon_size, beacon_indices = past_key_value

        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        # Raw projections (batch, seq, heads*head_dim)
        query_states, key_states, value_states = self.qkv_proj_with_beacon(
            hidden_states, beacon_size, beacon_indices
        )

        # Reshape to (batch, seq, heads, head_dim), apply q/k norm, then transpose
        query_states = self.q_norm(
            query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)

        key_states = self.k_norm(
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)  # (bsz, num_kv_heads, q_len, head_dim)

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_kv_heads, q_len, head_dim)

        # Cache keys/values *before* applying RoPE (same convention as qwen2 beacon)
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Apply rotary positional embeddings
        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3SdpaAttention(Qwen3Attention):
    """Qwen3 attention using torch.nn.functional.scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "Qwen3Model is using Qwen3SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` "
                "does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. '
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = q_len
        past_key, past_value, beacon_size, beacon_indices = past_key_value

        if past_key is not None:
            kv_seq_len += past_key.shape[2]

        query_states, key_states, value_states = self.qkv_proj_with_beacon(
            hidden_states, beacon_size, beacon_indices
        )

        query_states = self.q_norm(
            query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        key_states = self.k_norm(
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}"
                )

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        return attn_output, None, past_key_value


class Qwen3FlashAttention2(Qwen3Attention):
    """Qwen3 flash attention module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = q_len

        past_key, past_value, beacon_size, beacon_indices = past_key_value
        if past_key is not None:
            kv_seq_len += past_key.shape[2]

        query_states, key_states, value_states = self.qkv_proj_with_beacon(
            hidden_states, beacon_size, beacon_indices
        )

        query_states = self.q_norm(
            query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        key_states = self.k_norm(
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        # FlashAttention expects layout [batch, seq, heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.scaling,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length,
        dropout=0.0, softmax_scale=None
    ):
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout,
                softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer, key_layer, value_layer, indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


QWEN3_ATTENTION_CLASSES = {
    "eager": Qwen3Attention,
    "sdpa": Qwen3SdpaAttention,
    "flash_attention_2": Qwen3FlashAttention2,
}


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN3_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------

QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks).
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids`, pass an embedded representation.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# ---------------------------------------------------------------------------
# Qwen3Model
# ---------------------------------------------------------------------------

@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`].

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # BEACON: add beacon embedding
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.beacon_embed_tokens._is_hf_initialized = True

        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def _resolve_beacon_init_token_id(self):
        """Return a valid integer token id for beacon embedding init, or None to skip."""
        if self.config.beacon_embed_init == "bos":
            token_id = self.config.bos_token_id
            fallback = self.config.eos_token_id
        else:  # "eos"
            token_id = self.config.eos_token_id
            fallback = self.config.bos_token_id

        # Unwrap list (some configs store multiple ids)
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if isinstance(fallback, list):
            fallback = fallback[0] if fallback else None

        if token_id is None and fallback is not None:
            logger.warning_once(
                f"beacon_embed_init='{self.config.beacon_embed_init}' but corresponding token_id is None "
                f"in model config (it may live in generation_config.json). "
                f"Falling back to the other special token id ({fallback}) for beacon embedding init."
            )
            token_id = fallback

        if token_id is None:
            logger.warning_once(
                "Both bos_token_id and eos_token_id are None in model config. "
                "beacon_embed_tokens will keep its default random initialization."
            )
            return None

        return int(token_id)

    def _init_beacon_embed(self, missing_keys):
        """Initialize the beacon token embedding with that of the eos/bos token."""
        if is_deepspeed_zero3_enabled():
            import deepspeed
            params = [self.beacon_embed_tokens.weight, self.embed_tokens.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                # deepspeed will initialize the parameters to zero
                if (self.beacon_embed_tokens.weight == 0).all():
                    token_id = self._resolve_beacon_init_token_id()
                    if token_id is not None:
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[token_id]
        else:
            if any("beacon_embed_tokens" in missing_key for missing_key in missing_keys):
                token_id = self._resolve_beacon_init_token_id()
                if token_id is not None:
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[token_id]

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # BEACON: always use cache
        use_cache = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key, past_value, beacon_size, beacon_indices = past_key_values[0]

        # BEACON: separately embed ordinal tokens and beacon tokens
        if beacon_size > 0:
            is_beacon_token = input_ids >= self.config.vocab_size

            if is_beacon_token.any():
                max_beacon_id = self.config.vocab_size + self.beacon_embed_tokens.num_embeddings - 1
                bad_low = input_ids[is_beacon_token] < self.config.vocab_size
                bad_high = input_ids[is_beacon_token] > max_beacon_id
                if bad_low.any() or bad_high.any():
                    bad_ids = torch.unique(input_ids[is_beacon_token][bad_low | bad_high]).detach().cpu().tolist()
                    raise ValueError(
                        "Found out-of-range beacon token ids. "
                        f"Expected beacon ids in [{self.config.vocab_size}, {max_beacon_id}], "
                        f"but found {bad_ids}."
                    )

            safe_input_ids = input_ids.clamp(max=self.config.vocab_size - 1)
            ordinal_inputs_embeds = self.embed_tokens(safe_input_ids)

            beacon_embed_indices = (input_ids - self.config.vocab_size).clamp(
                min=0, max=self.beacon_embed_tokens.num_embeddings - 1
            )
            beacon_input_embeds = self.beacon_embed_tokens(beacon_embed_indices)

            inputs_embeds = torch.where(
                is_beacon_token.unsqueeze(-1), beacon_input_embeds, ordinal_inputs_embeds
            )
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # BEACON: still use tuple to organise cache
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ---------------------------------------------------------------------------
# Qwen3ForCausalLM
# ---------------------------------------------------------------------------

class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        if not hasattr(self, "generation_config") or self.generation_config is None:
            self.generation_config = GenerationConfig.from_model_config(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to initialise beacon parameters."""
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        config = model.config
        model.memory = Memory(
            model_config=config,
            k_seq_dim=2,
            v_seq_dim=2,
        )

        missing_keys = loading_info["missing_keys"]
        model.model._init_beacon_embed(missing_keys)
        for layer in model.model.layers:
            layer.self_attn._init_beacon_proj(missing_keys)

        return model

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None:
            # beacon_size=0 → no beacon parameters used
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        batch_loss = None
        token_loss = None

        if labels is not None:
            loss, batch_loss, token_loss = compute_loss(logits, labels, shift=False)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            token_loss=token_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _beacon_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        **kwargs,
    ):
        self.memory.prepare(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            skip_first=beacon_skip_first,
            skip_last=beacon_skip_last,
        )

        while not self.memory.finish:
            input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()

            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

            self.memory.update_memory(outputs.past_key_values)

            if labels is not None:
                self.memory.update_loss(outputs.batch_loss, (labels != -100).sum(-1))

        outputs = self.memory.output(outputs)
        return outputs

    def forward(self, **kwargs):
        """Forward computation over a batch of sequences."""
        with optional_grad_ctx(with_grad=self.training):
            if hasattr(self, "_enable_beacon") and self._enable_beacon is False:
                return self._native_forward(**kwargs)
            else:
                return self._beacon_forward(**kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
