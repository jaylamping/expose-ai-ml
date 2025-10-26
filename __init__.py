"""
Expose AI ML Library

A flexible ML library supporting PyTorch, ONNX, and LangChain with Google Cloud integration.
"""

__version__ = "0.1.0"
__author__ = "Joey Lamping"

from .core import MLFramework
from .llm import LLMChain
from .models import ModelRegistry
from .utils import DeviceManager

__all__ = [
    "MLFramework",
    "LLMChain", 
    "ModelRegistry",
    "DeviceManager",
]
