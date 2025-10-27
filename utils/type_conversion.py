"""
Type conversion utilities for API serialization.
"""
import numpy as np
from typing import Any, Dict, List, Union


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def safe_numpy_mean(values: List[float]) -> float:
    """
    Calculate mean of values, converting numpy types to native Python float.
    
    Args:
        values: List of numeric values
        
    Returns:
        Native Python float
    """
    if not values:
        return 0.0
    return float(np.mean(values))


def safe_numpy_std(values: List[float]) -> float:
    """
    Calculate standard deviation of values, converting numpy types to native Python float.
    
    Args:
        values: List of numeric values
        
    Returns:
        Native Python float
    """
    if not values or len(values) < 2:
        return 0.0
    return float(np.std(values))


def safe_numpy_max(values: List[float]) -> float:
    """
    Calculate maximum of values, converting numpy types to native Python float.
    
    Args:
        values: List of numeric values
        
    Returns:
        Native Python float
    """
    if not values:
        return 0.0
    return float(np.max(values))


def safe_numpy_min(values: List[float]) -> float:
    """
    Calculate minimum of values, converting numpy types to native Python float.
    
    Args:
        values: List of numeric values
        
    Returns:
        Native Python float
    """
    if not values:
        return 0.0
    return float(np.min(values))
