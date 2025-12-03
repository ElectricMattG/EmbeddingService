"""
Custom exception classes for the FastAPI Embedding Service.

These exceptions provide structured error handling with appropriate HTTP status codes
and detailed error messages for different failure scenarios.
"""

from typing import Any, Optional

from fastapi import HTTPException, status


class ModelNotLoadedException(HTTPException):
    """Exception raised when trying to use an embedding engine that hasn't been loaded."""

    def __init__(
        self,
        detail: str = "No embedding model is currently loaded. Please load a model first.",
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers=headers
        )


class InvalidModelException(HTTPException):
    """Exception raised when trying to load an invalid or unsupported model."""

    def __init__(
        self,
        model_name: str,
        available_models: Optional[list] = None,
        detail: Optional[str] = None,
        headers: Optional[dict] = None
    ):
        if detail is None:
            if available_models:
                detail = f"Invalid model '{model_name}'. Available models: {', '.join(available_models)}"
            else:
                detail = f"Invalid model '{model_name}'. Please check the model identifier."

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            headers=headers
        )


class EmbeddingProcessingException(HTTPException):
    """Exception raised when embedding processing fails."""

    def __init__(
        self,
        detail: str = "Failed to process embeddings. Please check your input and try again.",
        original_error: Optional[Exception] = None,
        headers: Optional[dict] = None
    ):
        if original_error:
            detail = f"{detail} Original error: {str(original_error)}"

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers
        )


class GPUMemoryException(HTTPException):
    """Exception raised when GPU memory is insufficient or operations fail."""

    def __init__(
        self,
        detail: str = "Insufficient GPU memory or GPU operation failed. Try reducing batch size or using CPU.",
        memory_info: Optional[dict] = None,
        headers: Optional[dict] = None
    ):
        if memory_info:
            used = memory_info.get('used_gb', 'unknown')
            total = memory_info.get('total_gb', 'unknown')
            detail = f"{detail} GPU memory: {used}GB used / {total}GB total"

        super().__init__(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=detail,
            headers=headers
        )


class ConcurrencyLimitException(HTTPException):
    """Exception raised when the service concurrency limit is exceeded."""

    def __init__(
        self,
        limit: Optional[int] = None,
        detail: Optional[str] = None,
        headers: Optional[dict] = None
    ):
        if detail is None:
            if limit:
                detail = f"Service is at maximum concurrency limit ({limit} requests). Please try again later."
            else:
                detail = "Service is at maximum concurrency limit. Please try again later."

        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers or {"Retry-After": "5"}
        )


class ConfigurationException(HTTPException):
    """Exception raised when there are configuration-related errors."""

    def __init__(
        self,
        detail: str = "Service configuration error. Please contact the administrator.",
        config_key: Optional[str] = None,
        headers: Optional[dict] = None
    ):
        if config_key:
            detail = f"Configuration error for '{config_key}': {detail}"

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers
        )


class ValidationException(HTTPException):
    """Exception raised for input validation errors."""

    def __init__(
        self,
        detail: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        headers: Optional[dict] = None
    ):
        if field:
            detail = f"Validation error for field '{field}': {detail}"
            if value is not None:
                detail = f"{detail} (received: {value})"

        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            headers=headers
        )


class ServiceUnavailableException(HTTPException):
    """Exception raised when the service is temporarily unavailable."""

    def __init__(
        self,
        detail: str = "Service is temporarily unavailable. Please try again later.",
        retry_after: int = 60,
        headers: Optional[dict] = None
    ):
        if headers is None:
            headers = {"Retry-After": str(retry_after)}
        else:
            headers["Retry-After"] = str(retry_after)

        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers=headers
        )


class ModelLoadingException(HTTPException):
    """Exception raised when model loading fails."""

    def __init__(
        self,
        model_name: str,
        detail: Optional[str] = None,
        original_error: Optional[Exception] = None,
        headers: Optional[dict] = None
    ):
        if detail is None:
            detail = f"Failed to load model '{model_name}'"

        if original_error:
            detail = f"{detail}. Error: {str(original_error)}"

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers
        )


class TimeoutException(HTTPException):
    """Exception raised when operations exceed timeout limits."""

    def __init__(
        self,
        operation: str = "operation",
        timeout_seconds: Optional[float] = None,
        detail: Optional[str] = None,
        headers: Optional[dict] = None
    ):
        if detail is None:
            if timeout_seconds:
                detail = f"The {operation} exceeded the timeout limit of {timeout_seconds} seconds"
            else:
                detail = f"The {operation} exceeded the timeout limit"

        super().__init__(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=detail,
            headers=headers
        )


# Utility functions for exception handling

def handle_gpu_error(error: Exception, operation: str = "GPU operation") -> GPUMemoryException:
    """Convert various GPU-related errors to GPUMemoryException."""
    error_msg = str(error).lower()

    if "out of memory" in error_msg or "cuda out of memory" in error_msg:
        return GPUMemoryException(
            detail=f"GPU out of memory during {operation}. Try reducing batch size or switching to CPU."
        )
    elif "cuda" in error_msg:
        return GPUMemoryException(
            detail=f"CUDA error during {operation}: {str(error)}"
        )
    else:
        return GPUMemoryException(
            detail=f"GPU operation failed during {operation}: {str(error)}"
        )


def handle_model_error(error: Exception, model_name: str, operation: str = "model operation") -> HTTPException:
    """Convert model-related errors to appropriate exceptions."""
    error_msg = str(error).lower()

    if "not found" in error_msg or "does not exist" in error_msg:
        return InvalidModelException(
            model_name=model_name,
            detail=f"Model '{model_name}' not found or does not exist"
        )
    elif "permission" in error_msg or "access" in error_msg:
        return ModelLoadingException(
            model_name=model_name,
            detail=f"Permission denied accessing model '{model_name}'",
            original_error=error
        )
    elif "memory" in error_msg or "out of memory" in error_msg:
        return handle_gpu_error(error, f"loading model '{model_name}'")
    else:
        return ModelLoadingException(
            model_name=model_name,
            detail=f"Failed to perform {operation} on model '{model_name}'",
            original_error=error
        )


def handle_processing_error(error: Exception, operation: str = "processing") -> HTTPException:
    """Convert processing errors to appropriate exceptions."""
    error_msg = str(error).lower()

    if "cuda" in error_msg or "gpu" in error_msg or "memory" in error_msg:
        return handle_gpu_error(error, operation)
    elif "timeout" in error_msg:
        return TimeoutException(operation=operation, detail=str(error))
    elif "validation" in error_msg or "invalid" in error_msg:
        return ValidationException(detail=str(error))
    else:
        return EmbeddingProcessingException(
            detail=f"Processing failed during {operation}",
            original_error=error
        )


def convert_to_http_exception(error: Exception, operation: str = "operation") -> HTTPException:
    """
    Convert any exception to an appropriate HTTPException.

    Args:
        error: The original exception
        operation: Description of the operation that failed

    Returns:
        HTTPException with appropriate status code and message
    """
    # If it's already an HTTPException, return it
    if isinstance(error, HTTPException):
        return error

    # Use the existing error handling logic
    return handle_processing_error(error, operation)
