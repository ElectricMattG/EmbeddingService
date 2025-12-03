"""
JSON serialization utilities for API responses.

This module provides utilities for converting numpy arrays and other data types
to JSON-serializable formats for FastAPI responses.
"""

from typing import Any, List, Union

import numpy as np


def numpy_to_list(array: np.ndarray) -> Union[List[float], List[List[float]]]:
    """
    Convert numpy array to list for JSON serialization.

    Args:
        array: Numpy array to convert

    Returns:
        List representation of the array:
        - 1D array: List[float]
        - 2D array: List[List[float]]

    Raises:
        ValueError: If array has more than 2 dimensions
    """
    if array.ndim == 1:
        return array.tolist()
    elif array.ndim == 2:
        return array.tolist()
    else:
        raise ValueError(f"Unsupported array dimensions: {array.ndim}. Only 1D and 2D arrays are supported.")


def numpy_matrix_to_list(array: np.ndarray) -> List[List[float]]:
    """
    Convert numpy array to 2D list for matrix-like JSON serialization.

    This function ensures the output is always a 2D list, wrapping 1D arrays
    in an outer list. Useful for similarity matrices and embedding batches.

    Args:
        array: Numpy array to convert

    Returns:
        2D list representation:
        - 1D array: [[...values...]]
        - 2D array: [[row1], [row2], ...]

    Raises:
        ValueError: If array has more than 2 dimensions
    """
    if array.ndim == 1:
        return [array.tolist()]
    elif array.ndim == 2:
        return array.tolist()
    else:
        raise ValueError(f"Unsupported array dimensions: {array.ndim}. Only 1D and 2D arrays are supported.")


def safe_float_conversion(value: Any) -> float:
    """
    Safely convert numpy scalars and other numeric types to Python float.

    Args:
        value: Value to convert

    Returns:
        Python float value

    Raises:
        ValueError: If value cannot be converted to float
    """
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert {type(value)} to float: {value}") from e


def validate_embedding_shape(embeddings: np.ndarray, expected_dims: int = None) -> None:
    """
    Validate embedding array shape and properties.

    Args:
        embeddings: Numpy array containing embeddings
        expected_dims: Expected embedding dimension (optional)

    Raises:
        ValueError: If embeddings have invalid shape or properties
    """
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(embeddings)}")

    if embeddings.ndim not in [1, 2]:
        raise ValueError(f"Embeddings must be 1D or 2D array, got {embeddings.ndim}D")

    if embeddings.size == 0:
        raise ValueError("Embeddings array is empty")

    if expected_dims is not None:
        actual_dims = embeddings.shape[-1] if embeddings.ndim == 2 else embeddings.shape[0]
        if actual_dims != expected_dims:
            raise ValueError(f"Expected {expected_dims} dimensions, got {actual_dims}")

    # Check for NaN or infinite values
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")
