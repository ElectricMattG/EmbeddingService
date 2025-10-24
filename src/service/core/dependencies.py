"""
FastAPI dependency injection system for EmbeddingService.

This module provides reusable dependencies for:
- EmbeddingEngine instance access
- Semaphore management for concurrency control
- Configuration access
- Input validation helpers
- Rate limiting and resource management
"""

import asyncio
import logging
from typing import Annotated, Dict, Any, Optional
from functools import lru_cache

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer

from .engine_manager import EmbeddingEngineManager
from .exceptions import (
    ConcurrencyLimitException,
    ModelNotLoadedException,
    convert_to_http_exception
)
from ..schemas.requests import (
    EmbedRequest,
    QuickEmbedRequest,
    SimilarityRequest,
    SearchRequest,
    ModelLoadRequest
)

# Initialize logger
logger = logging.getLogger(__name__)

# Optional security scheme for future authentication
security = HTTPBearer(auto_error=False)

# Semaphore configurations for different operation types
SEMAPHORE_LIMITS = {
    "gpu_operations": 8,     # GPU-intensive embedding operations (increased for 2-worker setup)
    "model_loading": 1,      # Only one model operation at a time
    "embedding_requests": 16, # General embedding requests (doubled for 2-worker setup)
    "similarity_requests": 12, # Similarity computations (doubled for 2-worker setup)
    "health_checks": 20      # Lightweight health operations
}

# Global semaphore storage
_semaphores: Dict[str, asyncio.Semaphore] = {}


def get_semaphore_limits() -> Dict[str, int]:
    """Get configured semaphore limits."""
    return SEMAPHORE_LIMITS.copy()


def initialize_semaphores() -> None:
    """Initialize all semaphores with configured limits."""
    global _semaphores
    for operation_type, limit in SEMAPHORE_LIMITS.items():
        _semaphores[operation_type] = asyncio.Semaphore(limit)
        logger.info(f"Initialized semaphore '{operation_type}' with limit {limit}")


def get_semaphore(operation_type: str = "embedding_requests") -> asyncio.Semaphore:
    """
    Get semaphore for specific operation type.
    
    Args:
        operation_type: Type of operation requesting semaphore
        
    Returns:
        Configured semaphore for the operation type
        
    Raises:
        ValueError: If operation_type is not recognized
    """
    if operation_type not in _semaphores:
        # Initialize if not exists (fallback for edge cases)
        if operation_type in SEMAPHORE_LIMITS:
            _semaphores[operation_type] = asyncio.Semaphore(SEMAPHORE_LIMITS[operation_type])
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
    
    return _semaphores[operation_type]


async def get_embedding_manager() -> EmbeddingEngineManager:
    """
    Dependency to get the singleton EmbeddingEngine manager.
    
    Returns:
        EmbeddingEngineManager instance
        
    Raises:
        HTTPException: If manager initialization fails
    """
    try:
        return EmbeddingEngineManager.get_instance()
    except Exception as e:
        logger.error(f"Failed to get embedding manager: {e}")
        raise convert_to_http_exception(e)


async def get_current_model_info(
    manager: Annotated[EmbeddingEngineManager, Depends(get_embedding_manager)]
) -> Dict[str, Any]:
    """
    Dependency to get current model information.
    
    Args:
        manager: EmbeddingEngine manager instance
        
    Returns:
        Current model information dictionary
        
    Raises:
        HTTPException: If no model is loaded
    """
    try:
        semaphore = get_semaphore("gpu_operations")
        async with semaphore:
            if not manager.is_model_loaded():
                raise ModelNotLoadedException("No model is currently loaded")
            
            engine = await manager.get_engine()
            return engine.get_model_info()
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise convert_to_http_exception(e)


async def validate_model_loaded(
    manager: Annotated[EmbeddingEngineManager, Depends(get_embedding_manager)]
) -> EmbeddingEngineManager:
    """
    Dependency to ensure a model is loaded before processing.
    
    Args:
        manager: EmbeddingEngine manager instance
        
    Returns:
        Manager instance if model is loaded
        
    Raises:
        HTTPException: If no model is loaded
    """
    if not manager.is_model_loaded():
        raise ModelNotLoadedException("No model is currently loaded. Please load a model first.")
    return manager


def validate_embed_request(request: EmbedRequest) -> EmbedRequest:
    """
    Validate embedding request parameters.
    
    Args:
        request: Embedding request to validate
        
    Returns:
        Validated request
        
    Raises:
        HTTPException: If validation fails
    """
    # Additional validation beyond Pydantic
    if len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="At least one text must be provided")
    
    # Check individual text lengths
    max_length = 8192  # Configurable limit
    for i, text in enumerate(request.texts):
        if len(text.strip()) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Text at index {i} is empty or whitespace only"
            )
        if len(text) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text at index {i} exceeds maximum length of {max_length} characters"
            )
    
    return request


def validate_quick_embed_request(request: QuickEmbedRequest) -> QuickEmbedRequest:
    """
    Validate quick embedding request parameters.
    
    Args:
        request: Quick embedding request to validate
        
    Returns:
        Validated request
        
    Raises:
        HTTPException: If validation fails
    """
    # Reuse basic validation
    base_request = EmbedRequest(
        texts=request.texts,
        batch_size=32,  # Default for quick requests
        show_progress=False,
        use_segmentation=request.use_segmentation
    )
    validate_embed_request(base_request)
    
    # Additional validation for model name if provided
    if request.model_name:
        from src.config import SMALL_MODELS
        if request.model_name not in SMALL_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name '{request.model_name}'. "
                       f"Available models: {list(SMALL_MODELS.keys())}"
            )
    
    return request


def validate_similarity_request(request: SimilarityRequest) -> SimilarityRequest:
    """
    Validate similarity computation request.
    
    Args:
        request: Similarity request to validate
        
    Returns:
        Validated request
        
    Raises:
        HTTPException: If validation fails
    """
    if len(request.texts1) == 0 or len(request.texts2) == 0:
        raise HTTPException(
            status_code=400, 
            detail="Both text lists must contain at least one text"
        )
    
    # Check batch size limits
    max_batch = 512  # Configurable limit
    if len(request.texts1) > max_batch or len(request.texts2) > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {max_batch} texts per list"
        )
    
    return request


def validate_search_request(request: SearchRequest) -> SearchRequest:
    """
    Validate similarity search request.
    
    Args:
        request: Search request to validate
        
    Returns:
        Validated request
        
    Raises:
        HTTPException: If validation fails
    """
    if len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    if len(request.candidates) == 0:
        raise HTTPException(
            status_code=400, 
            detail="At least one candidate text must be provided"
        )
    
    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")
    
    if request.top_k > len(request.candidates):
        raise HTTPException(
            status_code=400,
            detail="top_k cannot be greater than the number of candidates"
        )
    
    # Check candidate limits
    max_candidates = 1000  # Configurable limit
    if len(request.candidates) > max_candidates:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum number of candidates is {max_candidates}"
        )
    
    return request


def validate_model_load_request(request: ModelLoadRequest) -> ModelLoadRequest:
    """
    Validate model loading request.
    
    Args:
        request: Model load request to validate
        
    Returns:
        Validated request
        
    Raises:
        HTTPException: If validation fails
    """
    from src.config import SMALL_MODELS
    
    if request.model_name not in SMALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name '{request.model_name}'. "
                   f"Available models: {list(SMALL_MODELS.keys())}"
        )
    
    # Validate device
    valid_devices = ["auto", "cpu", "cuda", "mps"]
    if request.device not in valid_devices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{request.device}'. Valid options: {valid_devices}"
        )
    
    # Validate numeric parameters
    if request.max_length <= 0:
        raise HTTPException(status_code=400, detail="max_length must be greater than 0")
    
    if request.batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be greater than 0")
    
    return request


async def get_request_context(request: Request) -> Dict[str, Any]:
    """
    Extract useful context from the request for logging and monitoring.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dictionary with request context information
    """
    return {
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "content_type": request.headers.get("content-type"),
        "timestamp": asyncio.get_event_loop().time()
    }


async def check_concurrency_limit(
    operation_type: str = "embedding_requests",
    timeout: float = 30.0
) -> asyncio.Semaphore:
    """
    Check and acquire semaphore for operation with timeout.
    
    Args:
        operation_type: Type of operation requesting access
        timeout: Maximum time to wait for semaphore acquisition
        
    Returns:
        Acquired semaphore (to be used in async context manager)
        
    Raises:
        HTTPException: If semaphore cannot be acquired within timeout
    """
    semaphore = get_semaphore(operation_type)
    
    try:
        # Try to acquire with timeout
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        return semaphore
    except asyncio.TimeoutError:
        current_count = semaphore._value  # Available permits
        max_count = SEMAPHORE_LIMITS.get(operation_type, "unknown")
        
        raise ConcurrencyLimitException(
            f"Concurrency limit reached for {operation_type}. "
            f"Available: {current_count}/{max_count}. "
            f"Timeout after {timeout}s."
        )


@lru_cache()
def get_app_config() -> Dict[str, Any]:
    """
    Cached access to application configuration.
    
    Returns:
        Application configuration dictionary
    """
    try:
        from src.config import app_conf
        return app_conf
    except ImportError:
        logger.warning("Could not import app_conf, returning empty config")
        return {}


# Type annotations for dependency injection
EmbeddingManagerDep = Annotated[EmbeddingEngineManager, Depends(get_embedding_manager)]
ValidatedManagerDep = Annotated[EmbeddingEngineManager, Depends(validate_model_loaded)]
ModelInfoDep = Annotated[Dict[str, Any], Depends(get_current_model_info)]
RequestContextDep = Annotated[Dict[str, Any], Depends(get_request_context)]
AppConfigDep = Annotated[Dict[str, Any], Depends(get_app_config)]

# Validation dependencies
ValidEmbedRequestDep = Annotated[EmbedRequest, Depends(validate_embed_request)]
ValidQuickEmbedRequestDep = Annotated[QuickEmbedRequest, Depends(validate_quick_embed_request)]
ValidSimilarityRequestDep = Annotated[SimilarityRequest, Depends(validate_similarity_request)]
ValidSearchRequestDep = Annotated[SearchRequest, Depends(validate_search_request)]
ValidModelLoadRequestDep = Annotated[ModelLoadRequest, Depends(validate_model_load_request)]