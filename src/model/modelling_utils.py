import torch
from typing import Optional, Tuple

def move_padding_to(
        inputs: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        padding_side: str = "left"
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Move padding tokens to the front or end of the input_ids and attention_mask according to the attention mask.
        If labels are provided, they are also moved accordingly.

        Args:
            inputs: The inputs to move padding, which can be input ids or embeddings.
            attention_mask: The attention mask of the inputs.
            labels: Optional labels to move padding.
            padding_side: The side to move padding to. Can be 'left' or 'right'.

        Returns:
            The inputs with padding moved, the attention mask, and the labels.
                - inputs: The inputs with padding moved.
                - attention_mask: The attention mask of the inputs with padding moved.
                - labels: The labels with padding moved.
        """
        if padding_side == "left":
            # Sort by mask (0s first, then 1s) to move padding to left
            # stable=True maintains order of content tokens
            sorted_mask, indices = torch.sort(attention_mask, dim=1, descending=False, stable=True)
        elif padding_side == "right":
             # Sort by mask (1s first, then 0s) to move padding to right
            sorted_mask, indices = torch.sort(attention_mask, dim=1, descending=True, stable=True)
        else:
            raise ValueError(f"Invalid padding_side: {padding_side}")
        
        if inputs.dim() == 3:
            # In case that the iputs are embeddings of size [B, S, H], we need to expand the indices to [B, S, 1]
            # Expand indices for 3D gather: [B, S, H]
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, inputs.size(-1))
            inputs = torch.gather(inputs, 1, expanded_indices)
        else:
            inputs = torch.gather(inputs, 1, indices)
            
        attention_mask = sorted_mask
        
        if labels is not None:
            labels = torch.gather(labels, 1, indices)
            
        return inputs, attention_mask, labels

