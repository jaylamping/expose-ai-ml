"""
Device management utilities for ML models.
"""
import torch
from typing import Optional, Union
from config.settings import settings


class DeviceManager:
    """Manages device selection and model placement."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            device: Device to use ('cpu', 'cuda', 'auto', or None for settings default)
        """
        self.device = self._get_device(device or settings.device)
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate torch device."""
        import platform
        
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif platform.system() == "Darwin":
                # Force CPU on macOS to avoid MPS bus errors
                print("macOS detected - using CPU to avoid MPS bus errors")
                return torch.device("cpu")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")  # Apple Silicon (only if not macOS)
            else:
                return torch.device("cpu")
        else:
            # If explicitly requesting MPS on macOS, force CPU instead
            if device == "mps" and platform.system() == "Darwin":
                print("MPS requested on macOS - forcing CPU to avoid bus errors")
                return torch.device("cpu")
            return torch.device(device)
    
    def to_device(self, model_or_tensor: Union[torch.nn.Module, torch.Tensor]) -> Union[torch.nn.Module, torch.Tensor]:
        """Move model or tensor to the managed device."""
        return model_or_tensor.to(self.device)
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    def get_cuda_info(self) -> dict:
        """Get CUDA device information."""
        if not self.is_cuda_available():
            return {"available": False}
        
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
        }
