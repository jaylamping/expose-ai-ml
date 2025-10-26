"""
Main ML framework class that orchestrates PyTorch and ONNX models.
"""
import os
import torch
import onnxruntime as ort
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from .device_manager import DeviceManager
from config.settings import settings


class MLFramework:
    """
    Main framework for managing ML models with PyTorch and ONNX support.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the ML framework.
        
        Args:
            device: Device to use for models
        """
        self.device_manager = DeviceManager(device)
        self.models: Dict[str, Any] = {}
        self.onnx_sessions: Dict[str, ort.InferenceSession] = {}
        
        # Create model cache directory
        os.makedirs(settings.model_cache_dir, exist_ok=True)
    
    def load_pytorch_model(self, model_path: str, model_name: str, **kwargs) -> torch.nn.Module:
        """
        Load a PyTorch model.
        
        Args:
            model_path: Path to the model file
            model_name: Name to register the model under
            **kwargs: Additional arguments for torch.load
            
        Returns:
            Loaded PyTorch model
        """
        model = torch.load(model_path, map_location=self.device_manager.get_device(), **kwargs)
        
        if isinstance(model, torch.nn.Module):
            model = self.device_manager.to_device(model)
            model.eval()
        
        self.models[model_name] = model
        return model
    
    def register_pytorch_model(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """
        Register a PyTorch model directly.
        
        Args:
            model: PyTorch model to register
            model_name: Name to register the model under
            
        Returns:
            Registered PyTorch model
        """
        model = self.device_manager.to_device(model)
        model.eval()
        self.models[model_name] = model
        return model
    
    def load_onnx_model(self, model_path: str, model_name: str, providers: Optional[List[str]] = None) -> ort.InferenceSession:
        """
        Load an ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
            model_name: Name to register the model under
            providers: ONNX Runtime providers (default: auto-detect)
            
        Returns:
            ONNX Runtime inference session
        """
        if providers is None:
            providers = self._get_onnx_providers()
        
        session = ort.InferenceSession(model_path, providers=providers)
        self.onnx_sessions[model_name] = session
        return session
    
    def _get_onnx_providers(self) -> List[str]:
        """Get appropriate ONNX Runtime providers based on available hardware."""
        providers = ['CPUExecutionProvider']
        
        if self.device_manager.is_cuda_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        return providers
    
    def convert_pytorch_to_onnx(self, 
                              pytorch_model: torch.nn.Module, 
                              input_shape: tuple, 
                              output_path: str,
                              input_names: Optional[List[str]] = None,
                              output_names: Optional[List[str]] = None,
                              **kwargs) -> str:
        """
        Convert a PyTorch model to ONNX format.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Input tensor shape (batch_size, ...)
            output_path: Path to save the ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            **kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            Path to the saved ONNX model
        """
        pytorch_model.eval()
        
        # Move model to CPU for ONNX export (ONNX export works better with CPU)
        original_device = next(pytorch_model.parameters()).device
        pytorch_model_cpu = pytorch_model.to('cpu')
        dummy_input = torch.randn(input_shape)
        
        try:
            torch.onnx.export(
                pytorch_model_cpu,
                dummy_input,
                output_path,
                input_names=input_names or ['input'],
                output_names=output_names or ['output'],
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                **kwargs
            )
        finally:
            # Move model back to original device
            pytorch_model.to(original_device)
        
        return output_path
    
    def predict_pytorch(self, model_name: str, input_data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using a PyTorch model.
        
        Args:
            model_name: Name of the registered model
            input_data: Input tensor
            
        Returns:
            Prediction tensor
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        input_data = self.device_manager.to_device(input_data)
        
        with torch.no_grad():
            prediction = model(input_data)
        
        return prediction
    
    def predict_onnx(self, model_name: str, input_data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Make predictions using an ONNX model.
        
        Args:
            model_name: Name of the registered model
            input_data: Input data (tensor or numpy array)
            
        Returns:
            Prediction as numpy array
        """
        if model_name not in self.onnx_sessions:
            raise ValueError(f"ONNX model '{model_name}' not found")
        
        session = self.onnx_sessions[model_name]
        
        # Convert to numpy if needed
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        prediction = session.run(None, {input_name: input_data})
        
        return prediction[0]  # Return first output
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a registered model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        info = {"name": model_name}
        
        if model_name in self.models:
            model = self.models[model_name]
            info.update({
                "type": "pytorch",
                "device": str(self.device_manager.get_device()),
                "parameters": sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else None,
            })
        
        if model_name in self.onnx_sessions:
            session = self.onnx_sessions[model_name]
            info.update({
                "type": "onnx",
                "providers": session.get_providers(),
                "inputs": [{"name": inp.name, "shape": inp.shape, "type": inp.type} for inp in session.get_inputs()],
                "outputs": [{"name": out.name, "shape": out.shape, "type": out.type} for out in session.get_outputs()],
            })
        
        return info
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(set(self.models.keys()) | set(self.onnx_sessions.keys()))
