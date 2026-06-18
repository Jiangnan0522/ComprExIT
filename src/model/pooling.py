from abc import ABC, abstractmethod
from re import X
from typing import Tuple, Optional
import torch
import math
from torch import nn
import torch.nn.functional as F
from loguru import logger

def get_pooling_factory(pooling_method: str):
    """
    Return the pooling class (not an instance) by its `.name`.

    Note: poolings are organized in an inheritance tree (e.g. layerwise poolings inherit from
    `ChunkAttentionPooling` / `MultiHeadChunkAttentionPooling`). Using `Pooling.__subclasses__()`
    only returns *direct* subclasses, so we must search recursively.
    """

    def _iter_all_subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _iter_all_subclasses(sub)

    for subclass in _iter_all_subclasses(Pooling):
        if getattr(subclass, "name", None) == pooling_method:
            return subclass
    raise ValueError(f"Invalid pooling method: {pooling_method}")
    
    
class Pooling(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        return

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, backbone: nn.Module, tokenizer=None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class PoolingOutputs:
    "An extensible pooling output object"
    def __init__(self, pooled: torch.Tensor, pooled_mask: torch.Tensor, **kwargs):
        self.pooled = pooled
        self.pooled_mask = pooled_mask
        for arg, value in kwargs.items():
            setattr(self, arg, value)


class AvgPooling(Pooling):
    name = "avg"
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, backbone: nn.Module, tokenizer=None) -> PoolingOutputs:
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled = last_hidden_state.mean(dim=1, keepdim=True)
        # Create mask (all ones, since it's a single token)
        pooled_mask = torch.ones(pooled.shape[0], pooled.shape[1], device=pooled.device, dtype=torch.long)
        return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask)


class SlidingWindowPooling(Pooling):
    name = "sliding"
    def __init__(self, config):
        super().__init__(config)
        # You can add these fields to your Config class, or default them here
        # If stride < window_size, you get overlapping windows (more tokens)
        # If stride == window_size, you get non-overlapping blocks
        if (config.compression_ratio is not None):
            self.window_size = config.compression_ratio 
            self.stride = config.compression_ratio
            logger.info(f"Sliding window size: {self.window_size}, stride: {self.stride}")
        else:
            self.window_size = 4
            self.stride = 4
            logger.warning("Sliding window size and stride not set in the config, using default values (4, 4)")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, backbone: nn.Module, tokenizer=None) -> PoolingOutputs:
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [Batch, Seq, Hidden]
        
        # Transpose to [Batch, Hidden, Seq] for 1D pooling
        hidden_transposed = last_hidden_state.transpose(1, 2)  # [Batch, Hidden, Seq]
        
        # Build a mask over the sequence dimension so that padding tokens do not
        # contribute to the pooled representation.
        # attention_mask is typically [Batch, Seq] with 1 for real tokens and 0 for padding.
        if attention_mask is not None:
            seq_mask = attention_mask.unsqueeze(1).to(last_hidden_state.dtype)  # [Batch, 1, Seq]
        else:
            # Fallback: if no mask is provided, treat everything as a real token
            seq_mask = torch.ones(
                last_hidden_state.size(0),
                1,
                last_hidden_state.size(1),
                dtype=last_hidden_state.dtype,
                device=last_hidden_state.device,
            )
        
        # Zero out padding positions before pooling
        masked_hidden = hidden_transposed * seq_mask  # [Batch, Hidden, Seq]
        
        # We want the *sum* over valid tokens in each window divided by the
        # *number of valid tokens* in that window (masked average).
        # avg_pool1d gives us the mean over the full kernel_size, so we:
        #   1) pool the masked values and multiply by kernel_size -> sum over window
        #   2) pool the 0/1 mask and multiply by kernel_size      -> count of valid tokens
        pooled_sum = torch.nn.functional.avg_pool1d(
            masked_hidden,
            kernel_size=self.window_size,
            stride=self.stride,
            ceil_mode=True,
        ) * self.window_size  # [Batch, Hidden, Seq_Out]
        # Get the ratio of valid tokens in each window by pooling the mask
        pooled_mask_ratio = torch.nn.functional.avg_pool1d(
            seq_mask,
            kernel_size=self.window_size,
            stride=self.stride,
            ceil_mode=True,
        ) * self.window_size  # [Batch, 1, Seq_Out] = number of valid tokens

        # Generate the boolean mask for the pooled output.
        # If number of valid tokens > threshold (e.g. 0.1), then it's valid.
        # This handles the case where padding generates garbage windows at the beginning (left-padding)
        # or end (right-padding).
        # [Batch, 1, Seq_Out] -> [Batch, Seq_Out]
        output_mask = (pooled_mask_ratio.squeeze(1) > 0).long()

        # Avoid division by zero for windows that are entirely padding.
        pooled_mask_count = pooled_mask_ratio.clamp(min=1e-6)
        # Recover the values in each window by dividing the valid ratio
        pooled = pooled_sum / pooled_mask_count  # [Batch, Hidden, Seq_Out]

        pooled = pooled.transpose(1, 2)  # [Batch, Seq_Out, Hidden]

        if (self.config.add_global_avg is not None) and (self.config.add_global_avg):
            # Global masked average over the *sequence* dimension (ignoring padding)
            seq_mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [Batch, Seq, 1]
            masked_lhs = last_hidden_state * seq_mask_expanded  # [Batch, Seq, Hidden]
            sum_tokens = masked_lhs.sum(dim=1, keepdim=True)  # [Batch, 1, Hidden]
            token_counts = seq_mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [Batch, 1, 1]
            global_avg = sum_tokens / token_counts  # [Batch, 1, Hidden]

            pooled = torch.cat([pooled, global_avg], dim=1)
            # Add mask for global avg (always 1)
            global_mask = torch.ones(output_mask.shape[0], 1, device=output_mask.device, dtype=output_mask.dtype)
            output_mask = torch.cat([output_mask, global_mask], dim=1)
        
        # Transpose back to [Batch, Seq_Out, Hidden]
        return PoolingOutputs(pooled=pooled, pooled_mask=output_mask)


class SlidingWindowMaxPooling(Pooling):
    name = "sliding-max"
    def __init__(self, config):
        super().__init__(config)
        if (getattr(config, "compression_ratio", None) is not None):
            self.window_size = config.compression_ratio
            self.stride = config.compression_ratio
        else:
            self.window_size = 4
            self.stride = 4
            logger.warning("Compression ratio not set in the config, using default values (4, 4)")
        logger.info(f"Using sliding max pooling. Sliding window size: {self.window_size}, stride: {self.stride}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        backbone: nn.Module,
        tokenizer=None,
    ) -> PoolingOutputs:
        """
        Sliding-window max pooling over the sequence dimension.

        Padding-aware:
        - Padding tokens are excluded from the max via a large negative fill value.
        - Windows that contain only padding are marked invalid in the returned mask and their pooled
          vectors are set to 0 to avoid propagating -inf/-large-negative values.
        """
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError(f"window_size and stride must be > 0, got window_size={self.window_size}, stride={self.stride}")

        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [B, S, H]

        # [B, H, S] for 1D pooling
        hidden_transposed = last_hidden_state.transpose(1, 2)

        if attention_mask is not None:
            # [B, 1, S] float mask for pooling ops
            seq_mask = attention_mask.unsqueeze(1).to(dtype=hidden_transposed.dtype)
        else:
            seq_mask = torch.ones(
                last_hidden_state.size(0),
                1,
                last_hidden_state.size(1),
                dtype=hidden_transposed.dtype,
                device=hidden_transposed.device,
            )

        # Mask out padding tokens so they cannot win the max.
        # Use dtype-aware minimum finite value (works in fp16/bf16 without infs).
        min_val = torch.finfo(hidden_transposed.dtype).min
        masked_hidden = hidden_transposed.masked_fill(seq_mask == 0, min_val)  # [B, H, S]

        pooled = torch.nn.functional.max_pool1d(
            masked_hidden,
            kernel_size=self.window_size,
            stride=self.stride,
            ceil_mode=True,
        )  # [B, H, S_out]

        pooled_mask_float = torch.nn.functional.max_pool1d(
            seq_mask,
            kernel_size=self.window_size,
            stride=self.stride,
            ceil_mode=True,
        )  # [B, 1, S_out] = 1 if any valid token exists in window

        output_mask = pooled_mask_float.squeeze(1).to(dtype=torch.long)  # [B, S_out]

        # Zero out windows that are entirely padding, so we don't propagate min_val.
        pooled = pooled.masked_fill(output_mask.unsqueeze(1) == 0, 0.0)

        pooled = pooled.transpose(1, 2)  # [B, S_out, H]

        if (getattr(self.config, "add_global_avg", None) is not None) and (self.config.add_global_avg):
            # Global masked average over the *sequence* dimension (ignoring padding)
            if attention_mask is not None:
                seq_mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, S, 1]
            else:
                seq_mask_expanded = torch.ones(
                    last_hidden_state.size(0),
                    last_hidden_state.size(1),
                    1,
                    dtype=last_hidden_state.dtype,
                    device=last_hidden_state.device,
                )
            masked_lhs = last_hidden_state * seq_mask_expanded  # [B, S, H]
            sum_tokens = masked_lhs.sum(dim=1, keepdim=True)  # [B, 1, H]
            token_counts = seq_mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, 1]
            global_avg = sum_tokens / token_counts  # [B, 1, H]

            pooled = torch.cat([pooled, global_avg], dim=1)
            global_mask = torch.ones(output_mask.shape[0], 1, device=output_mask.device, dtype=output_mask.dtype)
            output_mask = torch.cat([output_mask, global_mask], dim=1)

        return PoolingOutputs(pooled=pooled, pooled_mask=output_mask)

class EchoPooling(Pooling):
    name = "echo"
    def __init__(self, config):
        super().__init__(config)
        
        self.echo_prompt_prefix = "Rewrite the sentence:"
        self.echo_prompt_suffix = "; The rewritten sentence:"

    def _add_echo_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer):
        if tokenizer is None:
            raise ValueError("EchoPooling requires a tokenizer in the forward pass.")
            
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Check if the first token is BOS token across the batch
        if tokenizer.bos_token_id is not None:
            has_bos = (input_ids[:, 0] == tokenizer.bos_token_id)
            if has_bos.all():
                 # remove the bos token first
                input_ids = input_ids[:, 1:]
                attention_mask = attention_mask[:, 1:]
            elif has_bos.any():
                # Mixed batch (some have BOS, some don't) - this is tricky.
                # Ideally we should handle per-sample. But for now, let's assume consistent batching.
                # Or we can just not remove it if not all have it?
                pass
        
        # Tokenize on the fly
        # We use the device of input_ids
        echo_tokenized_prefix = tokenizer(self.echo_prompt_prefix, return_tensors='pt', add_special_tokens=True) # add bos token
        echo_tokenized_suffix = tokenizer(self.echo_prompt_suffix, return_tensors='pt', add_special_tokens=False)
        
        echo_ids_prefix = echo_tokenized_prefix.input_ids.to(device).repeat(batch_size, 1)
        echo_attention_mask_prefix = echo_tokenized_prefix.attention_mask.to(device).repeat(batch_size, 1)
        echo_ids_suffix = echo_tokenized_suffix.input_ids.to(device).repeat(batch_size, 1)
        echo_attention_mask_suffix = echo_tokenized_suffix.attention_mask.to(device).repeat(batch_size, 1)

        input_ids = torch.cat(
            [
                echo_ids_prefix, 
                input_ids, 
                echo_ids_suffix,
                input_ids
            ], 
            dim=1
        )
        attention_mask = torch.cat(
            [
                echo_attention_mask_prefix, 
                attention_mask, 
                echo_attention_mask_suffix,
                attention_mask
            ], 
            dim=1
        )
        return input_ids, attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, backbone: nn.Module, tokenizer=None) -> PoolingOutputs:
        input_id_length = input_ids.shape[1]
        input_ids, attention_mask = self._add_echo_prompt(input_ids, attention_mask, tokenizer)
        last_hidden_state = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = last_hidden_state[:, -input_id_length:, :].mean(dim=1, keepdim=True)
        # Create mask (all ones)
        pooled_mask = torch.ones(pooled.shape[0], pooled.shape[1], device=pooled.device, dtype=torch.long)
        return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask)


class ConvPooling(Pooling):
    name = "conv"
    def __init__(self, config):
        super().__init__(config)
        
        # Default kernel size 3, stride 3 (non-overlapping chunks of 3)
        # You can adjust these in your config
        if (config.compression_ratio is not None):
            self.kernel_size = config.compression_ratio
            self.stride = config.compression_ratio
        else:
            self.kernel_size = 4
            self.stride = 4
            logger.warning("Compression ratio not set in the config, using default values (4, 4)")
        
        # Conv1d: in_channels=hidden_size, out_channels=hidden_size
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=True
        )
        # Optional: Initialize weights specifically if needed
        # nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, backbone: nn.Module, tokenizer=None) -> PoolingOutputs:
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [Batch, Seq, Hidden]
        
        # PyTorch Conv1d expects [Batch, Channels, Length]
        # So we swap Seq and Hidden dimensions
        x = last_hidden_state.transpose(1, 2) 
        
        if attention_mask is not None:
            # Mask input to avoid padding affecting the convolution
            # attention_mask: [Batch, Seq] -> [Batch, 1, Seq]
            mask_expanded = attention_mask.unsqueeze(1).to(dtype=x.dtype)
            x = x * mask_expanded

        # Apply Convolution
        # shape becomes: [Batch, Hidden, New_Seq_Len]
        pooled = self.conv(x)

        # Renormalize for partially-padded windows so activations are not
        # systematically shrunk when evaluation batches include padding.
        #
        # We compute the number of valid (non-pad) tokens per conv window via pooling
        # the 0/1 attention mask, then scale the *signal* part of the conv output by
        # (kernel_size / valid_count). Importantly, we do NOT scale the bias term.
        if attention_mask is not None:
            mask_float = attention_mask.unsqueeze(1).to(dtype=pooled.dtype)  # [B, 1, Seq]
            valid_count = torch.nn.functional.avg_pool1d(
                mask_float,
                kernel_size=self.kernel_size,
                stride=self.stride,
                ceil_mode=False,
            ) * float(self.kernel_size)  # [B, 1, New_Seq_Len]

            scale = (float(self.kernel_size) / valid_count.clamp(min=1.0)).to(dtype=pooled.dtype)  # [B, 1, L]

            if self.conv.bias is not None:
                bias = self.conv.bias.view(1, -1, 1)  # [1, Hidden, 1]
                pooled = (pooled - bias) * scale + bias
            else:
                pooled = pooled * scale
        
        # Swap back to [Batch, New_Seq_Len, Hidden]
        pooled = pooled.transpose(1, 2)
        
        if attention_mask is not None:
            # Pool the mask to match the new sequence length
            # We use max_pool1d to keep the window valid if there is at least one valid token
            mask_float = attention_mask.unsqueeze(1).float()
            pooled_mask_float = torch.nn.functional.max_pool1d(
                mask_float,
                kernel_size=self.kernel_size,
                stride=self.stride
            )
            pooled_mask = pooled_mask_float.squeeze(1).long()
        else:
            pooled_mask = torch.ones(pooled.shape[0], pooled.shape[1], device=pooled.device, dtype=torch.long)
        return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask)   


class ChunkAttentionPooling(Pooling):
    name = "chunk-attn"
    def __init__(self, config, return_attention_scores=False):
        super().__init__(config)

        self.chunk_size = getattr(config, "compression_ratio", 4)
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.return_attention_scores = return_attention_scores
        
        # q/k projection size (defaults to hidden_size if not configured)
        # - prefer an explicit chunk-attn setting if present
        # - otherwise reuse the layerwise gate hidden setting (used elsewhere in this repo)
        self.chunk_attn_hidden_size = getattr(config, "chunk_attn_hidden_size", None)
        if self.chunk_attn_hidden_size is None:
            self.chunk_attn_hidden_size = getattr(config, "layerwise_pooling_gate_hidden", None)
        if self.chunk_attn_hidden_size is None:
            self.chunk_attn_hidden_size = self.hidden_size
            logger.warning(
                f"chunk_attn_hidden_size not set in config; defaulting to hidden_size={self.hidden_size}"
            )
        self.chunk_attn_hidden_size = int(self.chunk_attn_hidden_size)

        # For q/k we use a (possibly) smaller hidden size. We do not project v.
        self.q_prj = nn.Linear(self.hidden_size, self.chunk_attn_hidden_size)
        self.k_prj = nn.Linear(self.hidden_size, self.chunk_attn_hidden_size)

        self.layer_norm_chunk_attn = nn.LayerNorm(self.hidden_size)
        logger.info(
            f"Chunk-attention pooling initialized with chunk size: {self.chunk_size}, "
            f"hidden size: {self.hidden_size}, qk hidden size: {self.chunk_attn_hidden_size}"
        )


    def _chunk_attention_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chunk-attention pooling over a sequence of hidden states.

        Args:
            hidden_states: [B, S, H]
            attention_mask: [B, S] with 1 for real tokens and 0 for padding, or None

        Returns:
            pooled: [B, num_chunks, H]
            pooled_mask: [B, num_chunks] (1 if chunk contains any real token, else 0)
            attention_scores: [B, num_chunks, chunk_size] (softmax weights per chunk)
        """

        batch_size, seq_len, hidden_size = hidden_states.shape

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if hidden_size != self.hidden_size:
            raise ValueError(f"Hidden size mismatch: got {hidden_size}, expected {self.hidden_size}")

        # Pad to a multiple of chunk_size so we can reshape safely.
        pad_tokens = (self.chunk_size - (seq_len % self.chunk_size)) % self.chunk_size
        if pad_tokens:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_tokens), value=0.0)  # pad seq dim on the right
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_tokens), value=0)

        # chunked_hidden_states: [batch, num_chunks, chunk_size, hidden]
        num_chunks = hidden_states.shape[1] // self.chunk_size
        chunked_hidden_states = hidden_states.view(batch_size, num_chunks, self.chunk_size, hidden_size)

        chunk_mask = None
        if attention_mask is not None:
            chunk_mask = attention_mask.view(batch_size, num_chunks, self.chunk_size).to(dtype=torch.bool)
        # v: do not normalize/project (keep original hidden space for pooling)
        v = chunked_hidden_states

        # layer norm for q/k computation
        x = self.layer_norm_chunk_attn(chunked_hidden_states)

        # Build a single query per chunk from a masked mean of tokens in that chunk.
        if chunk_mask is not None:
            denom = chunk_mask.sum(dim=2, keepdim=True).clamp(min=1).unsqueeze(-1) # [batch, num_chunks, 1, 1]
            x_masked = x * chunk_mask.unsqueeze(-1) # [batch, num_chunks, chunk_size, hidden]
            summary_states = x_masked.sum(dim=2, keepdim=True) / denom
        else:
            summary_states = x.mean(dim=2, keepdim=True)

        # q: [B, num_chunks, 1, Dh], k: [B, num_chunks, chunk_size, Dh]
        q = self.q_prj(summary_states)
        k = self.k_prj(x)

        # scores: [B, num_chunks, chunk_size]
        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(2) / math.sqrt(self.chunk_attn_hidden_size)

        if chunk_mask is not None:
            # mask out padding tokens before softmax according to current dtype
            scores = scores.masked_fill(~chunk_mask, torch.finfo(scores.dtype).min)

        # weights: [B, num_chunks, chunk_size]
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) # replace nan with 0

        # If we had padding, strictly (although they are already very small) zero out weights on padding positions and renormalize.
        if chunk_mask is not None:
            weights = weights * chunk_mask.to(dtype=weights.dtype)
            denom = weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
            weights = weights / denom

        # pooled: [batch, num_chunks, hidden]
        pooled = (weights.unsqueeze(-1) * v).sum(dim=2)

        # pooled_mask: [B, Tc] where a chunk is valid if it contains >=1 real token.
        if attention_mask is not None:
            pooled_mask = (chunk_mask.any(dim=2)).to(dtype=torch.long)
        else:
            pooled_mask = torch.ones(pooled.shape[0], pooled.shape[1], device=pooled.device, dtype=torch.long)
        return pooled, pooled_mask, weights

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        backbone: nn.Module, 
        tokenizer=None
    ) -> PoolingOutputs:

        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        pooled, pooled_mask, weights = self._chunk_attention_pool(hidden_states, attention_mask)

        if not self.return_attention_scores:
            return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask)
        else:
            return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask, attention_scores=weights)


class DynamicTokenLayerwiseChunkAttentionPooling(ChunkAttentionPooling):
    name = "dy-token-layer"    

    """
        Dynamically merge hidden states from **all layers** for each token.
    """

    def __init__(
        self,
        config,
        return_attention_scores: bool = False,
        return_layer_weights: bool = False,
        top_p_layers: float = None,
    ):
        super().__init__(config, return_attention_scores=return_attention_scores)

        self.return_layer_weights = return_layer_weights
        self._top_p_layers = top_p_layers

        # HF convention: hidden_states[0] is embedding output, then one per transformer block.
        num_hidden_layers = config.backbone_config_dict.get("num_hidden_layers", None)
        if num_hidden_layers is None:
            raise ValueError(
                "DynamicTokenLayerwiseChunkAttentionPooling requires `config.backbone_config_dict.num_hidden_layers` as a non-empty int."
            )
        num_hidden_layers = int(num_hidden_layers)
        self.num_layers = num_hidden_layers + 1  # including embedding output
        if getattr(self, "hidden_size", None) is None:
            raise ValueError("ChunkAttentionPooling did not set `self.hidden_size` as expected.")

        # learnable weights
        self.query_fusion_weights = nn.Parameter(torch.zeros(self.num_layers))
        # IMPORTANT: do NOT reuse/overwrite ChunkAttentionPooling's `q_prj` / `k_prj` attributes.
        # Those are used by `_chunk_attention_pool()` for chunk-attention pooling.
        self.layer_fusion_hidden_size = getattr(config, "layerwise_pooling_gate_hidden", None)
        if self.layer_fusion_hidden_size is None:
            self.layer_fusion_hidden_size = self.hidden_size
            logger.warning(f"layerwise_pooling_gate_hidden not set in the config, using default value of {self.hidden_size}")
        
        self.layer_fusion_hidden_size = int(self.layer_fusion_hidden_size)
        # Layer-fusion projections (separate from chunk-attention q/k projections).
        # q/k go to a smaller fusion space; v stays in full hidden space.
        self.layer_fusion_q_prj = nn.Linear(self.hidden_size, self.layer_fusion_hidden_size)
        self.layer_fusion_k_prj = nn.Linear(self.hidden_size, self.layer_fusion_hidden_size)
        self.layer_fusion_v_prj = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_fusion_layer_embeddings = nn.Parameter(torch.zeros(self.num_layers, self.layer_fusion_hidden_size))
        nn.init.normal_(self.layer_fusion_layer_embeddings, mean=0.0, std=0.02)

        self.temperature = getattr(config, "layerwise_pooling_temperature", 0.7)
        logger.info(
            "DynamicTokenLayerwiseChunkAttentionPooling initialized with "
            f"num_layers={self.num_layers}, layer_fusion_hidden_size={self.layer_fusion_hidden_size}, "
            f"temperature={self.temperature}"
        )

    @property
    def top_p_layers(self):
        return self._top_p_layers

    @top_p_layers.setter
    def top_p_layers(self, value: float):
        if not self.training:
            self._top_p_layers = value
            logger.info(f"Changing top-p layers for layer fusion to {value}")
        else:
            logger.warning(f"Top-p layers for layer fusion cannot be changed during training.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        backbone: nn.Module,
        tokenizer=None,
    ) -> PoolingOutputs:

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get the hidden states tuple of length K where each entry is [B, S, H]
        if isinstance(outputs, (tuple, list)):
            # HF tuple format when return_dict=False:
            hidden_states_tuple = outputs[2] if len(outputs) > 2 else None
        else:
            hidden_states_tuple = getattr(outputs, "hidden_states", None)
        
        if hidden_states_tuple is None:
            raise ValueError("Backbone did not return hidden_states. Make sure output_hidden_states=True is supported.")
        

        # Stack the layers at the 2nd dimension, to get [B, S, K, H]
        layer_stack = torch.stack(hidden_states_tuple, dim=2)

        # 1) Build a per-token query by a learned global mixture over layers.
        # query_fusion_probs: [K] -> broadcast to [B, S, K, 1]
        query_fusion_probs = torch.softmax(self.query_fusion_weights, dim=0).view(1, 1, self.num_layers, 1)
        query_fusions = (query_fusion_probs * layer_stack).sum(dim=2)  # [B, S, H]

        # 2) Cross-attend this token query to the K layer representations.
        q = self.layer_fusion_q_prj(query_fusions).unsqueeze(2)  # [B, S, 1, Dh]

        # k_raw: [B, S, K, Dh]; v: [B, S, K, H]
        k_raw = self.layer_fusion_k_prj(layer_stack) + self.layer_fusion_layer_embeddings.view(
            1, 1, self.num_layers, self.layer_fusion_hidden_size
        )
        k = k_raw.transpose(2, 3)  # [B, S, H, K]
        v = self.layer_fusion_v_prj(layer_stack)  # [B, S, K, H]

        layer_logits = torch.matmul(q, k) / math.sqrt(self.layer_fusion_hidden_size)  # [B, S, 1, K]
        scores = torch.softmax(layer_logits / self.temperature, dim=-1)  # [B, S, 1, K]
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Zero out padding tokens' layer weights (just for clarity and safety)
        if attention_mask is not None:
            token_mask = attention_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=scores.dtype)  # [B, S, 1, 1]
            scores = scores * token_mask

        # select top-p layers if not None
        if (self._top_p_layers is not None) and (not self.training):
            scores_sorted, indices_sorted = torch.sort(scores, dim=-1, descending=True)
            scores_accumulated = torch.cumsum(scores_sorted, dim=-1)
            
            # Mask out tokens with cumulative probability above the threshold
            # We want to keep tokens where cumsum <= p (plus the first one that crosses)
            sorted_indices_to_remove = scores_accumulated > self._top_p_layers
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter the sorted mask back to original indices
            indices_to_remove = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                dim=-1, index=indices_sorted, src=sorted_indices_to_remove
            )
            scores = scores.masked_fill(indices_to_remove, 0.0)
            # Renormalize
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        fused_hidden_states = (scores.transpose(2, 3) * v).sum(dim=2)  # [B, S, H]
        # NOTE: We do not handle paddings here yet because _chunk_attention_pool will handle it.

        pooled, pooled_mask, attn = self._chunk_attention_pool(fused_hidden_states, attention_mask)

        # return
        extras = {}
        if self.return_layer_weights:
            # Match TokenLayerwiseChunkAttentionPooling convention: [B, S, K]
            extras["layer_weights"] = scores.squeeze(2)
        if self.return_attention_scores:
            extras["attention_scores"] = attn

        return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask, **extras)


class OptimalTransportPooling(Pooling):
    name = "ot"

    """
    Optimal-Transport pooling (Version A):
    - Layer mix: reuse DynamicTokenLayerwiseChunkAttentionPooling-style token-level cross-attn over layers
    - Token merge: within each fixed window, compute OT plan from tokens -> anchors and aggregate values
    """

    def __init__(
        self,
        config,
        return_ot_plan: bool = False,
        return_layer_weights: bool = False,
    ):
        # NOTE: This pooling does NOT use chunk-attention pooling; it is a standalone pooling.
        super().__init__(config)

        self.hidden_size = getattr(config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError("OptimalTransportPooling requires `config.hidden_size` as a non-empty int.")
        self.hidden_size = int(self.hidden_size)

        self.return_ot_plan = return_ot_plan
        self.return_layer_weights = return_layer_weights

        # -----------------------------
        # 1) Layer mix (token-level cross-attn over layers) — same idea as DynamicTokenLayerwiseChunkAttentionPooling
        # -----------------------------
        if config.top_k_layers is not None and config.top_k_layers > 0:
            num_hidden_layers = config.top_k_layers
        else:
            num_hidden_layers = config.backbone_config_dict.get("num_hidden_layers", None)
            if num_hidden_layers is None:
                raise ValueError(
                    "OptimalTransportPooling requires `config.backbone_config_dict.num_hidden_layers` as a non-empty int."
                )
            num_hidden_layers = int(num_hidden_layers)
        self.num_layers = num_hidden_layers + 1  # including embedding output (HF hidden_states[0])

        # Global learned mixture to form per-token query (over layers)
        self.query_fusion_weights = nn.Parameter(torch.zeros(self.num_layers))

        # Projections for layer-fusion attention
        self.layer_fusion_hidden_size = getattr(config, "layerwise_pooling_gate_hidden", None)
        if self.layer_fusion_hidden_size is None:
            self.layer_fusion_hidden_size = self.hidden_size
            logger.warning(
                f"layerwise_pooling_gate_hidden not set in the config, using default value of {self.hidden_size}"
            )
        self.layer_fusion_hidden_size = int(self.layer_fusion_hidden_size)
        self.layer_fusion_q_prj = nn.Linear(self.hidden_size, self.layer_fusion_hidden_size)
        self.layer_fusion_k_prj = nn.Linear(self.hidden_size, self.layer_fusion_hidden_size)
        self.layer_fusion_v_prj = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_fusion_layer_embeddings = nn.Parameter(torch.zeros(self.num_layers, self.layer_fusion_hidden_size))
        nn.init.normal_(self.layer_fusion_layer_embeddings, mean=0.0, std=0.02)
        self.temperature = getattr(config, "layerwise_pooling_temperature", 0.7)

        # -----------------------------
        # 2) OT token merge (windowed)
        # -----------------------------
        self.window_size = getattr(config, "ot_window_size", None)
        if self.window_size is None:
            # Keep a reasonable default; chunk_size/compression_ratio in this repo is usually small (e.g. 4/8),
            # while OT windows are typically 64/128+.
            self.window_size = 128
            logger.warning("ot_window_size not set in config; defaulting to 128")
        self.window_size = int(self.window_size)
        if self.window_size <= 0:
            raise ValueError(f"ot_window_size must be > 0, got {self.window_size}")

        # OT compression ratio r and number of anchors per window K
        self.ot_ratio = int(getattr(config, "ot_compression_ratio", getattr(config, "compression_ratio", 4)))
        if self.ot_ratio <= 0:
            raise ValueError(f"ot_compression_ratio (or compression_ratio) must be > 0, got {self.ot_ratio}")

        # Number of anchors per window
        self.num_anchors = int(math.ceil(self.window_size / self.ot_ratio))

        # Cost weights and Sinkhorn hyperparams
        self.ot_alpha = float(getattr(config, "ot_alpha", 1.0))
        self.ot_beta = float(getattr(config, "ot_beta", 0.2))
        self.ot_eps = float(getattr(config, "ot_eps", 0.1))
        self.ot_n_iter = int(getattr(config, "ot_n_iter", 30))

        # Metric/value projections:
        # - metric head (Wc): used ONLY for semantic cost (cosine distance)
        # - value head (Wv): used ONLY for generating token values that get transported to anchors
        self.ot_metric_dim = getattr(config, "ot_metric_dim", None)
        if self.ot_metric_dim is None:
            self.ot_metric_dim = self.hidden_size
        self.ot_metric_dim = int(self.ot_metric_dim)
        if self.ot_metric_dim <= 0:
            raise ValueError(f"ot_metric_dim must be > 0, got {self.ot_metric_dim}")

        # parameters
        self.metric_ln = nn.LayerNorm(self.hidden_size)
        self.metric_prj = nn.Linear(self.hidden_size, self.ot_metric_dim, bias=True)  # Wc
        self.value_ln = nn.LayerNorm(self.hidden_size)
        self.value_prj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)     # Wv

        # Precompute segment assignment and positional cost
        seg_map, seg_centers = self._build_segment_map(self.window_size, self.num_anchors)
        self.register_buffer("ot_segment_map", seg_map, persistent=False)          # [W, K] float
        self.register_buffer("ot_segment_centers", seg_centers, persistent=False) # [K] float

        pos_i = torch.arange(self.window_size).float()  # [W]
        ## (i-c)^2 / (W^2)  - normalise by W^2 so that the upper bound of the cost is 1
        pos_cost = ((pos_i[:, None] - seg_centers[None, :]) ** 2) / float(self.window_size * self.window_size)  # [W,K]
        self.register_buffer("ot_pos_cost", pos_cost, persistent=False)

        logger.info(
            f"OptimalTransportPooling initialized with window_size={self.window_size}, "
            f"ot_ratio={self.ot_ratio}, num_anchors={self.num_anchors}, "
            f"ot_alpha={self.ot_alpha}, ot_beta={self.ot_beta}, ot_eps={self.ot_eps}, ot_n_iter={self.ot_n_iter}, "
            f"(num_layers={self.num_layers}, layer_fusion_hidden_size={self.layer_fusion_hidden_size})"
        )


    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    @staticmethod
    def _build_segment_map(window_size: int, num_anchors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build a [W,K] segment assignment matrix (0/1) and segment centers [K].
        Segments are nearly-equal partitions over positions [0..W-1].
        """
        if num_anchors <= 0 or window_size <= 0:
            raise ValueError(f"Invalid window_size={window_size}, num_anchors={num_anchors}")

        # Deterministic integer partition (no overlaps / no gaps).
        # If num_anchors > window_size, the extra anchors become empty (all zeros).
        base = window_size // num_anchors
        rem = window_size % num_anchors

        seg_map = torch.zeros(window_size, num_anchors, dtype=torch.float32)
        centers = torch.zeros(num_anchors, dtype=torch.float32)
        cur = 0
        for j in range(num_anchors):
            seg_len = base + (1 if j < rem else 0)
            s = cur
            e = min(window_size, cur + seg_len)
            cur = e
            if e > s:
                seg_map[s:e, j] = 1.0
                centers[j] = (float(s) + float(e - 1)) / 2.0
            else:
                # Empty segment: set a benign center (won't matter because b_j=0 and this column is masked)
                centers[j] = float(min(max(s, 0), window_size - 1))
        return seg_map, centers

    @staticmethod
    def sinkhorn_log(
        C: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        eps: float = 0.1,
        n_iter: int = 30,
        neg_inf: float = -1e9,
    ) -> torch.Tensor:
        """
        Log-domain Sinkhorn-Knopp with support for zero-mass rows/cols.

        Args:
            C: [N, W, K] cost
            a: [N, W] source mass (nonnegative, sum to 1 per N or all-zero)
            b: [N, K] target mass (nonnegative, sum to 1 per N or all-zero)
        Returns:
            P: [N, W, K] transport plan
        """
        C = C.float()
        a = a.float()
        b = b.float()

        N, W, K = C.shape
        logK = -C / float(eps)  # [N,W,K]

        row_ok = a > 0
        col_ok = b > 0

        loga = torch.where(row_ok, torch.log(a.clamp_min(1e-20)), torch.full_like(a, neg_inf))
        logb = torch.where(col_ok, torch.log(b.clamp_min(1e-20)), torch.full_like(b, neg_inf))

        u = torch.zeros(N, W, device=C.device, dtype=torch.float32)
        v = torch.zeros(N, K, device=C.device, dtype=torch.float32)

        for _ in range(int(n_iter)):
            # u = loga - logsumexp(logK + v)
            t = torch.logsumexp(logK + v[:, None, :], dim=2)  # [N,W]
            u_new = loga - t
            u = torch.where(row_ok, u_new, torch.full_like(u_new, neg_inf))

            # v = logb - logsumexp(logK + u)
            t = torch.logsumexp(logK + u[:, :, None], dim=1)  # [N,K]
            v_new = logb - t
            v = torch.where(col_ok, v_new, torch.full_like(v_new, neg_inf))

        logP = logK + u[:, :, None] + v[:, None, :]
        P = torch.exp(logP)
        # Explicitly zero-out invalid rows/cols
        P = P * row_ok[:, :, None].float() * col_ok[:, None, :].float()
        return P

    def _layer_fuse(self, hidden_states_tuple, attention_mask: Optional[torch.Tensor]):
        """
        Fuse hidden states across layers per token, returning:
        - fused_hidden_states: [B,S,H]
        - layer_weights: [B,S,K] (optional)
        """
        # Stack all layers: [B, S, K, H]
        layer_stack = torch.stack(hidden_states_tuple, dim=2)
        B, S, K, H = layer_stack.shape
        if K != self.num_layers:
            raise ValueError(f"Expected num_layers={self.num_layers}, got hidden_states_tuple length={K}")

        # Query: a learned global mixture over layers -> [B,S,H]
        query_fusion_probs = torch.softmax(self.query_fusion_weights, dim=0).view(1, 1, self.num_layers, 1)
        query_fusions = (query_fusion_probs * layer_stack).sum(dim=2)  # [B,S,H]

        # Cross-attend this token query to the K layer representations.
        q = self.layer_fusion_q_prj(query_fusions).unsqueeze(2)  # [B,S,1,Dh]

        k_raw = self.layer_fusion_k_prj(layer_stack) + self.layer_fusion_layer_embeddings.view(
            1, 1, self.num_layers, self.layer_fusion_hidden_size
        )  # [B,S,K,Dh]
        k = k_raw.transpose(2, 3)  # [B,S,Dh,K]
        v = self.layer_fusion_v_prj(layer_stack)  # [B,S,K,H]

        layer_logits = torch.matmul(q, k) / math.sqrt(self.layer_fusion_hidden_size)  # [B,S,1,K]
        scores = torch.softmax(layer_logits / self.temperature, dim=-1)  # [B,S,1,K]
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if attention_mask is not None:
            token_mask = attention_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=scores.dtype)  # [B,S,1,1]
            scores = scores * token_mask

        fused_hidden_states = (scores.transpose(2, 3) * v).sum(dim=2)  # [B,S,H]
        return fused_hidden_states, scores.squeeze(2)  # [B,S,K]

    def _ot_pool(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """FP32 OT pooling
        Computes the optimal transport plan between tokens and anchors using the sinkhorn algorithm.

        dimension simples:
        B: batch size
        T: number of tokens
        H: hidden size
        W: window size
        K: number of anchors
        nW: number of windows

        Args:
            x: [B,T,H] fused token representations
            attention_mask: [B,T] (1=valid,0=pad) or None
        Returns:
            z: [B, nW*K, H]
            z_mask: [B, nW*K]
            ot_plan (optional): [B, nW, W, K]
        """

        # Keep feature tensors in the model/original dtype (often bf16),
        # but keep OT "mass/probability" tensors in fp32 for numerical stability.
        orig_dtype = x.dtype

        B, T, H = x.shape
        W = self.window_size
        K = self.num_anchors

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=x.device, dtype=torch.long)


        # Initialize the mass of source tokens from attention mask (1 for valid tokens, 0 for padding).
        # NOTE: keep masses in fp32 even if the model runs in bf16/fp16.
        m = attention_mask.to(dtype=torch.float32)  # [B,T]

        nW = self._ceil_div(T, W)
        T_pad = nW * W
        pad_tokens = T_pad - T # number of tokens to pad to fullfill the window size
        if pad_tokens:
            x = F.pad(x, (0, 0, 0, pad_tokens), value=0.0)
            m = F.pad(m, (0, pad_tokens), value=0.0)

        xw = x.view(B, nW, W, H)  # [B,nW,W,H] (orig_dtype)
        mw = m.view(B, nW, W)     # [B,nW,W]   (fp32)


        # ---- 3) Initialize anchors by segmented masked mean (vectorized via segment_map)
        seg_map = self.ot_segment_map.to(device=x.device, dtype=torch.float32)  # [W,K] fp32
        # weights: [B,nW,W,K]
        seg_weights = mw[:, :, :, None] * seg_map[None, None, :, :]
        denom = seg_weights.sum(dim=2)  # Get the correct denom (when with paddings) for mean pooling[B,nW,K]
        seg_valid = denom > 0
        denom_safe = denom.clamp_min(1e-6)

        # numerator: [B,nW,K,H]
        # mean aggregation over the window dimension to get the anchor for each anchor
        # seg_weights: [B,nW,W,K] -> transpose to [B,nW,K,W]
        # xw:          [B,nW,W,H]
        # result:      [B,nW,K,H]
        # Compute anchors in fp32 for stability, but cast back to orig_dtype before LN/Linear modules.
        anchor_sum = torch.matmul(seg_weights.transpose(2, 3), xw.float())  # [B,nW,K,H] fp32
        anchors = (anchor_sum / denom_safe[:, :, :, None]).to(dtype=orig_dtype)  # [B,nW,K,H]

        # ---- 4) Cost: semantic (cosine) + positional
        # Metric embeddings (u_i, u_j) in fp32 for stability:
        # u_i = normalize(Wc(LN(x_i)))   -> [B,nW,W,dc]
        # u_j = normalize(Wc(LN(anchor_j))) -> [B,nW,K,dc]
        u_i = self.metric_prj(self.metric_ln(xw))
        u_j = self.metric_prj(self.metric_ln(anchors))
        u_i = F.normalize(u_i, dim=-1)
        u_j = F.normalize(u_j, dim=-1)

        # Cosine similarity as the semantic cost - FP32 for stability
        # u_i: [B,nW,W,dc], u_j: [B,nW,K,dc] => u_j^T: [B,nW,dc,K] => sim: [B,nW,W,K]
        sem_sim = torch.matmul(u_i.float(), u_j.float().transpose(-1, -2))
        sem_cost = 1.0 - sem_sim  # [B,nW,W,K]

        # pos_cost = self.ot_pos_cost.to(device=x.device, dtype=torch.float32)  # [W,K]
        # C = self.ot_alpha * sem_cost + self.ot_beta * pos_cost[None, None, :, :]  # [B,nW,W,K]
        C = sem_cost

        # ---- 5) get the masses of source tokens(a) and anchors(b) (support zeros)
        a_raw = mw  # [B,nW,W]
        a_sum = a_raw.sum(dim=2, keepdim=True)  # [B,nW,1]
        a = a_raw / a_sum.clamp_min(1.0)        # all-zero windows stay all-zero

        b_raw = seg_valid.to(dtype=torch.float32)  # [B,nW,K]
        b_sum = b_raw.sum(dim=2, keepdim=True)     # [B,nW,1]
        b = b_raw / b_sum.clamp_min(1.0)

        # ---- 5) Sinkhorn (log-domain) to solve the OT plan
        N = B * nW
        Cn = C.view(N, W, K)
        an = a.view(N, W)
        bn = b.view(N, K)
        Pn = self.sinkhorn_log(Cn, an, bn, eps=self.ot_eps, n_iter=self.ot_n_iter)  # [N,W,K]
        P = Pn.view(B, nW, W, K) # fp32

        # ---- 6) Apply plan to values
        v_tokens = self.value_prj(self.value_ln(xw))  # [B,nW,W,H]
        # Aggregate in fp32 then cast back
        # Aggregate values with the OT plan (P^T @ V):
        # P: [B,nW,W,K] -> transpose to [B,nW,K,W]
        # V: [B,nW,W,H]
        # Z: [B,nW,K,H]
        z = torch.matmul(P.to(dtype=torch.float32).transpose(2, 3), v_tokens.float())
        z = z / b.clamp_min(1e-6)[:, :, :, None]  # normalize by capacity
        z = z.to(dtype=orig_dtype) # cast back to the original dtype

        z_flat = z.reshape(B, nW * K, H)
        z_mask = seg_valid.reshape(B, nW * K).to(dtype=torch.long)
        return z_flat, z_mask, P

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        backbone: nn.Module,
        tokenizer=None,
    ) -> PoolingOutputs:
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if isinstance(outputs, (tuple, list)):
            hidden_states_tuple = outputs[2] if len(outputs) > 2 else None
        else:
            hidden_states_tuple = getattr(outputs, "hidden_states", None)
        if hidden_states_tuple is None:
            raise ValueError("Backbone did not return hidden_states. Make sure output_hidden_states=True is supported.")

        fused, layer_weights = self._layer_fuse(hidden_states_tuple, attention_mask)
        pooled, pooled_mask, ot_plan = self._ot_pool(fused, attention_mask)

        extras = {}
        if self.return_layer_weights:
            extras["layer_weights"] = layer_weights
        if self.return_ot_plan:
            extras["ot_plan"] = ot_plan

        return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask, **extras)




class SourceMass(nn.Module):
    """
    Compute learned source mass a for OT within each window.
    """
    def __init__(self, d, tau=1.0, lam_floor=0.05):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        # 2 layer mlp to learn the source mass
        self.mass_prj = nn.Linear(d, 1)
        self.tau = tau
        self.lam_floor = lam_floor

        # init mass_mlp to near-uniform at start
        nn.init.zeros_(self.mass_prj.weight)
        if self.mass_prj.bias is not None:
            nn.init.zeros_(self.mass_prj.bias)

    def forward(self, xw, mw):
        """
        xw: [B, nW, W, d]
        mw: [B, nW, W]  (bool or 0/1)
        return a: [B, nW, W] (sum over W == 1 for each window with any valid token)
        """
        B, nW, W, d = xw.shape
        mw_f = mw.float()

        # scores
        s = self.mass_prj(self.ln(xw)).squeeze(-1)            # [B,nW,W]

        # mask pads to -inf for softmax
        neg_inf = -1e9
        s = s.masked_fill(mw == 0, neg_inf)

        # importance distribution
        p = torch.softmax(s / self.tau, dim=-1)        # [B,nW,W] (pad positions ~0)

        # uniform over valid tokens
        denom = mw_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,nW,1]
        u = mw_f / denom                                        # [B,nW,W]

        # mix with floor and uniform distribution u, for stability (avoid early collapse)
        a = (1.0 - self.lam_floor) * p + self.lam_floor * u

        # enforce pad=0 and renorm
        a = a * mw_f
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        
        # Handle empty windows (where mw is all 0)
        valid_win = (mw_f.sum(dim=-1) > 0)  # [B,nW]
        a = torch.where(valid_win[:,:,None], a, torch.zeros_like(a))

        return a


class OptimalTransportDynamicSourcePooling(OptimalTransportPooling):
    name = "ot-dy-src"

    """
    Optimal-Transport pooling with dynamic source tokens:
    - Layer mix: reuse DynamicTokenLayerwiseChunkAttentionPooling-style token-level cross-attn over layers
    - Token merge: within each fixed window, compute OT plan from tokens -> anchors and aggregate values
    """

    def __init__(
        self,
        config,
        return_ot_plan: bool = False,
        return_layer_weights: bool = False,
    ):
        super().__init__(config, return_ot_plan, return_layer_weights)
        
        self.ab_ot_shuffle_anchors = getattr(config, "ab_ot_shuffle_anchors", False)
        if self.ab_ot_shuffle_anchors:
            logger.info(f"OptimalTransportDynamicSourcePooling initialized with ab_ot_shuffle_anchors=True")
        
        # Add SourceMass
        self.source_mass = SourceMass(
            d=self.hidden_size, 
            tau=getattr(config, "ot_source_tau", 1.0),
            lam_floor=getattr(config, "ot_source_lam_floor", 0.05)
        )

    def _ot_pool(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """FP32 OT pooling
        Computes the optimal transport plan between tokens and anchors using the sinkhorn algorithm.

        dimension simples:
        B: batch size
        T: number of tokens
        H: hidden size
        W: window size
        K: number of anchors
        nW: number of windows

        Args:
            x: [B,T,H] fused token representations
            attention_mask: [B,T] (1=valid,0=pad) or None
        Returns:
            z: [B, nW*K, H]
            z_mask: [B, nW*K]
            ot_plan (optional): [B, nW, W, K]
        """

        # Keep feature tensors in the model/original dtype (often bf16),
        # but keep OT "mass/probability" tensors in fp32 for numerical stability.
        orig_dtype = x.dtype

        B, T, H = x.shape
        W = self.window_size
        K = self.num_anchors

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=x.device, dtype=torch.long)


        # Initialize the mass of source tokens from attention mask (1 for valid tokens, 0 for padding).
        # NOTE: keep masses in fp32 even if the model runs in bf16/fp16.
        m = attention_mask.to(dtype=torch.float32)  # [B,T]

        nW = self._ceil_div(T, W)
        T_pad = nW * W
        pad_tokens = T_pad - T
        if pad_tokens:
            x = F.pad(x, (0, 0, 0, pad_tokens), value=0.0)
            m = F.pad(m, (0, pad_tokens), value=0.0)

        xw = x.view(B, nW, W, H)  # [B,nW,W,H] (orig_dtype)
        mw = m.view(B, nW, W)     # [B,nW,W]   (fp32)


        # ---- 3) Initialize anchors by segmented masked mean (vectorized via segment_map)
        seg_map = self.ot_segment_map.to(device=x.device, dtype=torch.float32)  # [W,K] fp32
        # weights: [B,nW,W,K]
        seg_weights = mw[:, :, :, None] * seg_map[None, None, :, :]
        denom = seg_weights.sum(dim=2)  # Get the correct denom (when with paddings) for mean pooling[B,nW,K]
        seg_valid = denom > 0
        denom_safe = denom.clamp_min(1e-6)

        # numerator: [B,nW,K,H]
        # mean aggregation over the window dimension to get the anchor for each anchor
        # seg_weights: [B,nW,W,K] -> transpose to [B,nW,K,W]
        # xw:          [B,nW,W,H]
        # result:      [B,nW,K,H]
        # Compute anchors in fp32 for stability, but cast back to orig_dtype before LN/Linear modules.
        anchor_sum = torch.matmul(seg_weights.transpose(2, 3), xw.float())  # [B,nW,K,H] fp32
        anchors = (anchor_sum / denom_safe[:, :, :, None]).to(dtype=orig_dtype)  # [B,nW,K,H]

        # Ablation: shuffle anchor order (along K) without changing how anchors are computed.
        # We also shuffle seg_valid correspondingly to keep all per-anchor tensors aligned.
        if self.ab_ot_shuffle_anchors:
            perm = torch.rand(B, nW, K, device=anchors.device).argsort(dim=-1)  # [B,nW,K]
            anchors = anchors.gather(2, perm[:, :, :, None].expand(-1, -1, -1, H))
            seg_valid = seg_valid.gather(2, perm)


        # ---- 4) Cost: semantic (cosine) + positional
        # Metric embeddings (u_i, u_j) in fp32 for stability:
        # u_i = normalize(Wc(LN(x_i)))   -> [B,nW,W,dc]
        # u_j = normalize(Wc(LN(anchor_j))) -> [B,nW,K,dc]
        u_i = self.metric_prj(self.metric_ln(xw))
        u_j = self.metric_prj(self.metric_ln(anchors))
        u_i = F.normalize(u_i, dim=-1)
        u_j = F.normalize(u_j, dim=-1)

        # Cosine similarity as the semantic cost - FP32 for stability
        # u_i: [B,nW,W,dc], u_j: [B,nW,K,dc] => u_j^T: [B,nW,dc,K] => sim: [B,nW,W,K]
        sem_sim = torch.matmul(u_i.float(), u_j.float().transpose(-1, -2))
        sem_cost = 1.0 - sem_sim  # [B,nW,W,K]

        # pos_cost = self.ot_pos_cost.to(device=x.device, dtype=torch.float32)  # [W,K]
        # C = self.ot_alpha * sem_cost + self.ot_beta * pos_cost[None, None, :, :]  # [B,nW,W,K]
        C = sem_cost

        # ---- 5) get the masses of source tokens(a) and anchors(b) (support zeros)
        # Modified: learnable source mass a
        a = self.source_mass(xw, mw) # [B, nW, W]

        b_raw = seg_valid.to(dtype=torch.float32)  # [B,nW,K]
        b_sum = b_raw.sum(dim=2, keepdim=True)     # [B,nW,1]
        b = b_raw / b_sum.clamp_min(1.0)

        # ---- 5) Sinkhorn (log-domain) to solve the OT plan
        N = B * nW
        Cn = C.view(N, W, K)
        an = a.view(N, W)
        bn = b.view(N, K)
        Pn = self.sinkhorn_log(Cn, an, bn, eps=self.ot_eps, n_iter=self.ot_n_iter)  # [N,W,K]
        P = Pn.view(B, nW, W, K) # fp32

        # ---- 6) Apply plan to values
        v_tokens = self.value_prj(self.value_ln(xw))  # [B,nW,W,H]
        # Aggregate in fp32 then cast back
        # Aggregate values with the OT plan (P^T @ V):
        # P: [B,nW,W,K] -> transpose to [B,nW,K,W]
        # V: [B,nW,W,H]
        # Z: [B,nW,K,H]
        z = torch.matmul(P.to(dtype=torch.float32).transpose(2, 3), v_tokens.float())
        z = z / b.clamp_min(1e-6)[:, :, :, None]  # normalize by capacity
        z = z.to(dtype=orig_dtype) # cast back to the original dtype

        z_flat = z.reshape(B, nW * K, H)
        z_mask = seg_valid.reshape(B, nW * K).to(dtype=torch.long)
        return z_flat, z_mask, P



class OptimalTransportDynamicSourceSingleLayerPooling(Pooling):
    name = "ot-dy-src-single-layer"

    """
    Layerwise: Use features from a single layer
    Tokenwise: OT with dynamic source tokens
    """


    def __init__(
        self,
        config,
        return_ot_plan: bool = False,
        return_layer_weights: bool = False,
    ):
        # NOTE: This pooling does NOT use chunk-attention pooling; it is a standalone pooling.
        super().__init__(config)

        self.hidden_size = getattr(config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError("OptimalTransportPooling requires `config.hidden_size` as a non-empty int.")
        self.hidden_size = int(self.hidden_size)

        self.return_ot_plan = return_ot_plan
        self.return_layer_weights = return_layer_weights

        # Ablation: shuffle anchor order (along K) without changing how anchors are computed.
        # (Matches `ot-dy-src` behavior.)
        self.ab_ot_shuffle_anchors = getattr(config, "ab_ot_shuffle_anchors", False)
        if self.ab_ot_shuffle_anchors:
            logger.info(f"{self.__class__.__name__} initialized with ab_ot_shuffle_anchors=True")

        # -----------------------------
        # 1) Layer mix
        # -----------------------------
        num_hidden_layers = config.backbone_config_dict.get("num_hidden_layers", None)
        if num_hidden_layers is None:
            raise ValueError(
                "OptimalTransportPooling requires `config.backbone_config_dict.num_hidden_layers` as a non-empty int."
            )
        num_hidden_layers = int(num_hidden_layers)
        self.num_layers = num_hidden_layers + 1  # including embedding output (HF hidden_states[0])
        self.layer_fusion_prj = nn.Linear(self.hidden_size, self.hidden_size)
        self.temperature = getattr(config, "layerwise_pooling_temperature", 0.7)
        # Use the 10th layer by default (embedding layer is the 0th layer).
        self.layer_index_selected = int(getattr(config, "layer_index_selected", -1))
        logger.info(f"OptimalTransportDynamicSourceSingleLayerPooling initialized with layer_index_selected={self.layer_index_selected}")

        # -----------------------------
        # 2) OT token merge (windowed)
        # -----------------------------
        self.window_size = getattr(config, "ot_window_size", None)
        if self.window_size is None:
            # Keep a reasonable default; chunk_size/compression_ratio in this repo is usually small (e.g. 4/8),
            # while OT windows are typically 64/128+.
            self.window_size = 128
            logger.warning("ot_window_size not set in config; defaulting to 128")
        self.window_size = int(self.window_size)
        if self.window_size <= 0:
            raise ValueError(f"ot_window_size must be > 0, got {self.window_size}")

        # OT compression ratio r and number of anchors per window K
        self.ot_ratio = int(getattr(config, "ot_compression_ratio", getattr(config, "compression_ratio", 4)))
        if self.ot_ratio <= 0:
            raise ValueError(f"ot_compression_ratio (or compression_ratio) must be > 0, got {self.ot_ratio}")

        # Number of anchors per window
        self.num_anchors = int(math.ceil(self.window_size / self.ot_ratio))

        # Cost weights and Sinkhorn hyperparams
        self.ot_alpha = float(getattr(config, "ot_alpha", 1.0))
        self.ot_beta = float(getattr(config, "ot_beta", 0.2))
        self.ot_eps = float(getattr(config, "ot_eps", 0.1))
        self.ot_n_iter = int(getattr(config, "ot_n_iter", 30))

        # Metric/value projections:
        # - metric head (Wc): used ONLY for semantic cost (cosine distance)
        # - value head (Wv): used ONLY for generating token values that get transported to anchors
        self.ot_metric_dim = getattr(config, "ot_metric_dim", None)
        if self.ot_metric_dim is None:
            self.ot_metric_dim = self.hidden_size
        self.ot_metric_dim = int(self.ot_metric_dim)
        if self.ot_metric_dim <= 0:
            raise ValueError(f"ot_metric_dim must be > 0, got {self.ot_metric_dim}")

        # parameters
        self.metric_ln = nn.LayerNorm(self.hidden_size)
        self.metric_prj = nn.Linear(self.hidden_size, self.ot_metric_dim, bias=True)  # Wc
        self.value_ln = nn.LayerNorm(self.hidden_size)
        self.value_prj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)     # Wv

        # Precompute segment assignment and positional cost
        seg_map, seg_centers = self._build_segment_map(self.window_size, self.num_anchors)
        self.register_buffer("ot_segment_map", seg_map, persistent=False)          # [W, K] float
        self.register_buffer("ot_segment_centers", seg_centers, persistent=False) # [K] float

        pos_i = torch.arange(self.window_size).float()  # [W]
        ## (i-c)^2 / (W^2)  - normalise by W^2 so that the upper bound of the cost is 1
        pos_cost = ((pos_i[:, None] - seg_centers[None, :]) ** 2) / float(self.window_size * self.window_size)  # [W,K]
        self.register_buffer("ot_pos_cost", pos_cost, persistent=False)

        # Add SourceMass
        self.source_mass = SourceMass(
            d=self.hidden_size, 
            tau=getattr(config, "ot_source_tau", 1.0),
            lam_floor=getattr(config, "ot_source_lam_floor", 0.05)
        )

        logger.info(
            f"OptimalTransportDynamicSourceSingleLayerPooling initialized with window_size={self.window_size}, "
            f"ot_ratio={self.ot_ratio}, num_anchors={self.num_anchors}, "
            f"ot_alpha={self.ot_alpha}, ot_beta={self.ot_beta}, ot_eps={self.ot_eps}, ot_n_iter={self.ot_n_iter}, "
            f"(num_layers={self.num_layers})"
        )


    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    @staticmethod
    def _build_segment_map(window_size: int, num_anchors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build a [W,K] segment assignment matrix (0/1) and segment centers [K].
        Segments are nearly-equal partitions over positions [0..W-1].
        """
        if num_anchors <= 0 or window_size <= 0:
            raise ValueError(f"Invalid window_size={window_size}, num_anchors={num_anchors}")

        # Deterministic integer partition (no overlaps / no gaps).
        # If num_anchors > window_size, the extra anchors become empty (all zeros).
        base = window_size // num_anchors
        rem = window_size % num_anchors

        seg_map = torch.zeros(window_size, num_anchors, dtype=torch.float32)
        centers = torch.zeros(num_anchors, dtype=torch.float32)
        cur = 0
        for j in range(num_anchors):
            seg_len = base + (1 if j < rem else 0)
            s = cur
            e = min(window_size, cur + seg_len)
            cur = e
            if e > s:
                seg_map[s:e, j] = 1.0
                centers[j] = (float(s) + float(e - 1)) / 2.0
            else:
                # Empty segment: set a benign center (won't matter because b_j=0 and this column is masked)
                centers[j] = float(min(max(s, 0), window_size - 1))
        return seg_map, centers

    @staticmethod
    def sinkhorn_log(
        C: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        eps: float = 0.1,
        n_iter: int = 30,
        neg_inf: float = -1e9,
    ) -> torch.Tensor:
        """
        Log-domain Sinkhorn-Knopp with support for zero-mass rows/cols.

        Args:
            C: [N, W, K] cost
            a: [N, W] source mass (nonnegative, sum to 1 per N or all-zero)
            b: [N, K] target mass (nonnegative, sum to 1 per N or all-zero)
        Returns:
            P: [N, W, K] transport plan
        """
        C = C.float()
        a = a.float()
        b = b.float()

        N, W, K = C.shape
        logK = -C / float(eps)  # [N,W,K]

        row_ok = a > 0
        col_ok = b > 0

        loga = torch.where(row_ok, torch.log(a.clamp_min(1e-20)), torch.full_like(a, neg_inf))
        logb = torch.where(col_ok, torch.log(b.clamp_min(1e-20)), torch.full_like(b, neg_inf))

        u = torch.zeros(N, W, device=C.device, dtype=torch.float32)
        v = torch.zeros(N, K, device=C.device, dtype=torch.float32)

        for _ in range(int(n_iter)):
            # u = loga - logsumexp(logK + v)
            t = torch.logsumexp(logK + v[:, None, :], dim=2)  # [N,W]
            u_new = loga - t
            u = torch.where(row_ok, u_new, torch.full_like(u_new, neg_inf))

            # v = logb - logsumexp(logK + u)
            t = torch.logsumexp(logK + u[:, :, None], dim=1)  # [N,K]
            v_new = logb - t
            v = torch.where(col_ok, v_new, torch.full_like(v_new, neg_inf))

        logP = logK + u[:, :, None] + v[:, None, :]
        P = torch.exp(logP)
        # Explicitly zero-out invalid rows/cols
        P = P * row_ok[:, :, None].float() * col_ok[:, None, :].float()
        return P


    def _ot_pool(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """FP32 OT pooling
        Computes the optimal transport plan between tokens and anchors using the sinkhorn algorithm.

        dimension simples:
        B: batch size
        T: number of tokens
        H: hidden size
        W: window size
        K: number of anchors
        nW: number of windows

        Args:
            x: [B,T,H] fused token representations
            attention_mask: [B,T] (1=valid,0=pad) or None
        Returns:
            z: [B, nW*K, H]
            z_mask: [B, nW*K]
            ot_plan (optional): [B, nW, W, K]
        """

        # Keep feature tensors in the model/original dtype (often bf16),
        # but keep OT "mass/probability" tensors in fp32 for numerical stability.
        orig_dtype = x.dtype

        B, T, H = x.shape
        W = self.window_size
        K = self.num_anchors

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=x.device, dtype=torch.long)


        # Initialize the mass of source tokens from attention mask (1 for valid tokens, 0 for padding).
        # NOTE: keep masses in fp32 even if the model runs in bf16/fp16.
        m = attention_mask.to(dtype=torch.float32)  # [B,T]

        nW = self._ceil_div(T, W)
        T_pad = nW * W
        pad_tokens = T_pad - T
        if pad_tokens:
            x = F.pad(x, (0, 0, 0, pad_tokens), value=0.0)
            m = F.pad(m, (0, pad_tokens), value=0.0)

        xw = x.view(B, nW, W, H)  # [B,nW,W,H] (orig_dtype)
        mw = m.view(B, nW, W)     # [B,nW,W]   (fp32)


        # ---- 3) Initialize anchors by segmented masked mean (vectorized via segment_map)
        seg_map = self.ot_segment_map.to(device=x.device, dtype=torch.float32)  # [W,K] fp32
        # weights: [B,nW,W,K]
        seg_weights = mw[:, :, :, None] * seg_map[None, None, :, :]
        denom = seg_weights.sum(dim=2)  # Get the correct denom (when with paddings) for mean pooling[B,nW,K]
        seg_valid = denom > 0
        denom_safe = denom.clamp_min(1e-6)

        # numerator: [B,nW,K,H]
        # mean aggregation over the window dimension to get the anchor for each anchor
        # seg_weights: [B,nW,W,K] -> transpose to [B,nW,K,W]
        # xw:          [B,nW,W,H]
        # result:      [B,nW,K,H]
        # Compute anchors in fp32 for stability, but cast back to orig_dtype before LN/Linear modules.
        anchor_sum = torch.matmul(seg_weights.transpose(2, 3), xw.float())  # [B,nW,K,H] fp32
        anchors = (anchor_sum / denom_safe[:, :, :, None]).to(dtype=orig_dtype)  # [B,nW,K,H]

        # Ablation: shuffle anchor order (along K) without changing how anchors are computed.
        # We also shuffle seg_valid correspondingly to keep all per-anchor tensors aligned.
        if self.ab_ot_shuffle_anchors:
            perm = torch.rand(B, nW, K, device=anchors.device).argsort(dim=-1)  # [B,nW,K]
            anchors = anchors.gather(2, perm[:, :, :, None].expand(-1, -1, -1, H))
            seg_valid = seg_valid.gather(2, perm)


        # ---- 4) Cost: semantic (cosine) + positional
        # Metric embeddings (u_i, u_j) in fp32 for stability:
        # u_i = normalize(Wc(LN(x_i)))   -> [B,nW,W,dc]
        # u_j = normalize(Wc(LN(anchor_j))) -> [B,nW,K,dc]
        u_i = self.metric_prj(self.metric_ln(xw))
        u_j = self.metric_prj(self.metric_ln(anchors))
        u_i = F.normalize(u_i, dim=-1)
        u_j = F.normalize(u_j, dim=-1)

        # Cosine similarity as the semantic cost - FP32 for stability
        # u_i: [B,nW,W,dc], u_j: [B,nW,K,dc] => u_j^T: [B,nW,dc,K] => sim: [B,nW,W,K]
        sem_sim = torch.matmul(u_i.float(), u_j.float().transpose(-1, -2))
        sem_cost = 1.0 - sem_sim  # [B,nW,W,K]

        # pos_cost = self.ot_pos_cost.to(device=x.device, dtype=torch.float32)  # [W,K]
        # C = self.ot_alpha * sem_cost + self.ot_beta * pos_cost[None, None, :, :]  # [B,nW,W,K]
        C = sem_cost

        # ---- 5) get the masses of source tokens(a) and anchors(b) (support zeros)
        # Modified: learnable source mass a
        a = self.source_mass(xw, mw) # [B, nW, W]

        b_raw = seg_valid.to(dtype=torch.float32)  # [B,nW,K]
        b_sum = b_raw.sum(dim=2, keepdim=True)     # [B,nW,1]
        b = b_raw / b_sum.clamp_min(1.0)

        # ---- 5) Sinkhorn (log-domain) to solve the OT plan
        N = B * nW
        Cn = C.view(N, W, K)
        an = a.view(N, W)
        bn = b.view(N, K)
        Pn = self.sinkhorn_log(Cn, an, bn, eps=self.ot_eps, n_iter=self.ot_n_iter)  # [N,W,K]
        P = Pn.view(B, nW, W, K) # fp32

        # ---- 6) Apply plan to values
        v_tokens = self.value_prj(self.value_ln(xw))  # [B,nW,W,H]
        # Aggregate in fp32 then cast back
        # Aggregate values with the OT plan (P^T @ V):
        # P: [B,nW,W,K] -> transpose to [B,nW,K,W]
        # V: [B,nW,W,H]
        # Z: [B,nW,K,H]
        z = torch.matmul(P.to(dtype=torch.float32).transpose(2, 3), v_tokens.float())
        z = z / b.clamp_min(1e-6)[:, :, :, None]  # normalize by capacity
        z = z.to(dtype=orig_dtype) # cast back to the original dtype

        z_flat = z.reshape(B, nW * K, H)
        z_mask = seg_valid.reshape(B, nW * K).to(dtype=torch.long)
        return z_flat, z_mask, P

    def _layer_fuse(self, hidden_states_tuple, attention_mask: Optional[torch.Tensor]):
        """
        Only use one single layer for layerfuse.
        """
        if hidden_states_tuple is None:
            raise ValueError("hidden_states_tuple is None")

        K = len(hidden_states_tuple)
        if K != self.num_layers:
            raise ValueError(f"Expected num_layers={self.num_layers}, got hidden_states_tuple length={K}")

        # Support negative indices like Python.
        idx = int(self.layer_index_selected)
        # Select one layer then project as OT input representation.
        x = hidden_states_tuple[idx]  # [B, S, H]
        fused_hidden_states = self.layer_fusion_prj(x)  # [B, S, H]

        # For compatibility with other OT poolings' debug outputs: return per-token one-hot weights over layers.
        B, S, _ = fused_hidden_states.shape
        layer_weights = torch.zeros(B, S, self.num_layers, device=fused_hidden_states.device, dtype=torch.float32)
        layer_weights[:, :, idx] = 1.0
        if attention_mask is not None:
            layer_weights = layer_weights * attention_mask.unsqueeze(-1).to(dtype=layer_weights.dtype)

        return fused_hidden_states, layer_weights

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            backbone: nn.Module,
            tokenizer=None,
        ) -> PoolingOutputs:
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            if isinstance(outputs, (tuple, list)):
                hidden_states_tuple = outputs[2] if len(outputs) > 2 else None
            else:
                hidden_states_tuple = getattr(outputs, "hidden_states", None)
            if hidden_states_tuple is None:
                raise ValueError("Backbone did not return hidden_states. Make sure output_hidden_states=True is supported.")

            fused, layer_weights = self._layer_fuse(hidden_states_tuple, attention_mask)
            pooled, pooled_mask, ot_plan = self._ot_pool(fused, attention_mask)

            extras = {}
            if self.return_layer_weights:
                extras["layer_weights"] = layer_weights
            if self.return_ot_plan:
                extras["ot_plan"] = ot_plan

            return PoolingOutputs(pooled=pooled, pooled_mask=pooled_mask, **extras)
