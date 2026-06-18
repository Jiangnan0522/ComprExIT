import importlib.util
import torch
from typing import Optional
try:
    import intel_extension_for_pytorch as ipex  # ensures xpu backend is registered
except ImportError:
    ipex = None


def supports_flash_attention_2(device: Optional[int] = None) -> bool:
    # A supported GPU is not enough -- the `flash_attn` package must also be importable.
    # `uv sync` does not install flash-attn (it is only required by the Activation Beacon
    # baseline), so on the default environment we report False and let callers fall back.
    if importlib.util.find_spec("flash_attn") is None:
        return False
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return major >= 8  # Ampere+ (SM80+)

def get_device_module():
    # Determine device module
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device_module = torch.xpu
        device_type = "xpu"
    elif torch.cuda.is_available():
        device_module = torch.cuda
        device_type = "cuda"
    else:
        device_module = None
        device_type = "cpu"
    return device_module, device_type

def get_device_stats(device_index=0):
    """
    Get device (XPU or GPU) memory statistics for a specific device.
    
    Args:
        device_index: Index of the device (default: 0)
        
    Returns:
        dict: Dictionary containing memory statistics
    """
    stats = {}
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        num_devices = torch.xpu.device_count()
        device = torch.device(f"xpu:{device_index}")
        props = torch.xpu.get_device_properties(device_index)
        total_memory_gib = props.total_memory / 1024**3

        allocated = torch.xpu.memory_allocated(device)
        reserved = torch.xpu.memory_reserved(device)
        allocated_gib = allocated / 1024**3
        reserved_gib = reserved / 1024**3

        mem_stats = torch.xpu.memory_stats(device)
        peak_usage_gib = mem_stats['allocated_bytes.all.peak'] / 1024**3
        
        stats = {
            'device_type': 'xpu',
            'num_devices': round(num_devices, 2),
            'total_memory_gib': round(total_memory_gib, 2),
            'allocated_gib': round(allocated_gib, 2),
            'reserved_gib': round(reserved_gib, 2),
            'peak_usage_gib': round(peak_usage_gib, 2)
        }
    elif torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device = torch.device(f"cuda:{device_index}")
        props = torch.cuda.get_device_properties(device_index)
        total_memory_gib = props.total_memory / 1024**3

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        allocated_gib = allocated / 1024**3
        reserved_gib = reserved / 1024**3

        # CUDA memory_stats returns a dict with many keys. 'allocated_bytes.all.peak' is consistent.
        mem_stats = torch.cuda.memory_stats(device)
        peak_usage_gib = mem_stats.get('allocated_bytes.all.peak', 0) / 1024**3
        
        stats = {
            'device_type': 'cuda',
            'num_devices': round(num_devices, 2),
            'total_memory_gib': round(total_memory_gib, 2),
            'allocated_gib': round(allocated_gib, 2),
            'reserved_gib': round(reserved_gib, 2),
            'peak_usage_gib': round(peak_usage_gib, 2)
        }
        
    return stats

# Backward compatibility alias
xpu_stats = get_device_stats


    