"""Device management utilities for GPU/CUDA support."""

import os
import logging
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# Global device cache
_DEVICE_CACHE = None


def get_device(device: Optional[str] = None, prefer_gpu: bool = True) -> str:
    """
    Get or determine the compute device to use.
    
    Priority:
    1. Explicit device parameter
    2. SYNTHETIC_DATA_DEVICE environment variable
    3. Auto-detection based on hardware availability
    
    Args:
        device: Explicit device string (e.g., 'cpu', 'cuda', 'cuda:0')
        prefer_gpu: If True and device is None, prefer GPU if available
    
    Returns:
        Device string suitable for PyTorch/TensorFlow
        
    Raises:
        ValueError: If specified device is invalid
    """
    global _DEVICE_CACHE
    
    # Use explicit parameter if provided
    if device is not None:
        if not _is_valid_device(device):
            raise ValueError(f"Invalid device: {device}")
        return device
    
    # Check environment variable
    env_device = os.environ.get('SYNTHETIC_DATA_DEVICE')
    if env_device:
        if not _is_valid_device(env_device):
            LOGGER.warning(f"Invalid device from SYNTHETIC_DATA_DEVICE: {env_device}, falling back to default")
        else:
            return env_device
    
    # Auto-detect
    if prefer_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def set_device_env(device: str) -> None:
    """
    Set the global device via environment variable.
    
    Args:
        device: Device string (e.g., 'cpu', 'cuda', 'cuda:0')
        
    Raises:
        ValueError: If device is invalid
    """
    if not _is_valid_device(device):
        raise ValueError(f"Invalid device: {device}")
    os.environ['SYNTHETIC_DATA_DEVICE'] = device
    global _DEVICE_CACHE
    _DEVICE_CACHE = None  # Clear cache


def _is_valid_device(device: str) -> bool:
    """Check if a device string is valid by attempting to create a test tensor."""
    if not TORCH_AVAILABLE:
        return device == 'cpu'
    try:
        torch.tensor([1.0], device=device)
        return True
    except (RuntimeError, ValueError):
        return False


def validate_and_get_device(device: Optional[str] = None) -> tuple[str, bool]:
    """
    Validate device and determine if it's GPU.
    
    Args:
        device: Device string or None
        
    Returns:
        Tuple of (device_string, is_gpu_bool)
    """
    selected_device = get_device(device)
    is_gpu = _is_valid_device('cuda') and ('cuda' in selected_device)
    return selected_device, is_gpu


def to_device(tensor, device: Optional[str] = None):
    """
    Move a tensor or list of tensors to specified device.
    
    Args:
        tensor: PyTorch tensor or list of tensors
        device: Device string or None (uses default)
        
    Returns:
        Tensor(s) on specified device
    """
    if device is None:
        device = get_device()
    
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(to_device(t, device) for t in tensor)
    
    if hasattr(tensor, 'to'):
        return tensor.to(device)
    
    return tensor
