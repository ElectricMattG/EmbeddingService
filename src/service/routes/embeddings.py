"""
Embedding generation routes for the FastAPI Embedding Service.

Provides endpoints for generating text embeddings with support for batching,
segmentation, and different processing modes.
"""

import time
import numpy as np
from typing import List, Union, Tuple
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from src.service.core.engine_manager import embedding_manager
from src.service.schemas.requests import EmbedRequest, QuickEmbedRequest
from src.service.schemas.responses import EmbedResponse, EmbedWithSegmentsResponse
from src.service.core.exceptions import (
    ModelNotLoadedException,
    InvalidModelException,
    EmbeddingProcessingException,
    GPUMemoryException,
    ConcurrencyLimitException,
    ValidationException
)
from src.utils.convenience import quick_embed, quick_embed_with_segments
from src.service.utils.serialization import numpy_to_list, validate_embedding_shape
from src.service.utils.validation import validate_text_input, validate_batch_size

# Create router
router = APIRouter(tags=["Embedding Generation"])


@router.post(
    "",
    response_model=EmbedResponse,
    summary="Generate embeddings",
    description="Generate embeddings for input texts using the currently loaded model with batching and segmentation support.",
    responses={
        200: {
            "description": "Embeddings generated successfully",
            "model": EmbedResponse
        },
        422: {
            "description": "Invalid input validation",
            "content": {
                "application/json": {
                    "example": {
                        "error": "ValidationError",
                        "message": "Text input is too long",
                        "details": {"max_length": 100000, "received_length": 150000}
                    }
                }
            }
        },
        429: {
            "description": "Too many concurrent requests"
        },
        503: {
            "description": "No model is currently loaded"
        },
        507: {
            "description": "Insufficient GPU memory"
        },
        500: {
            "description": "Embedding processing failed"
        }
    }
)
async def generate_embeddings(request: EmbedRequest) -> EmbedResponse:
    """
    Generate embeddings for input texts with automatic batching and segmentation.
    
    This endpoint uses the currently loaded model to generate embeddings with support for:
    - Single text strings or lists of texts
    - Configurable batch sizes for processing efficiency
    - Automatic text segmentation for long documents
    - Progress tracking for large batches
    
    Args:
        request: Embedding request with texts and optional parameters
    
    Returns:
        EmbedResponse: Generated embeddings with shape and processing time
    
    Raises:
        ModelNotLoadedException: If no model is currently loaded
        ValidationException: If input validation fails
        EmbeddingProcessingException: If embedding generation fails
        GPUMemoryException: If GPU memory is insufficient
        ConcurrencyLimitException: If service is at capacity
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        validate_text_input(request.texts, "texts")
        if request.batch_size is not None:
            validate_batch_size(request.batch_size, "batch_size")
        
        # Generate embeddings using the manager
        embeddings = await embedding_manager.embed(
            texts=request.texts,
            batch_size=request.batch_size,
            show_progress=request.show_progress,
            use_segmentation=request.use_segmentation
        )
        
        # Validate and convert numpy array to list for JSON serialization
        validate_embedding_shape(embeddings)
        embeddings_list = numpy_to_list(embeddings)
        processing_time = time.time() - start_time
        
        return EmbedResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape),
            processing_time=processing_time
        )
        
    except (ModelNotLoadedException, ValidationException, EmbeddingProcessingException, 
            GPUMemoryException, ConcurrencyLimitException):
        raise
    except ValueError as e:
        raise ValidationException(detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error generating embeddings: {str(e)}"
        )


@router.post(
    "/with-segments",
    response_model=EmbedWithSegmentsResponse,
    summary="Generate embeddings with segmentation info",
    description="Generate embeddings and return detailed information about text segmentation used for long documents.",
    responses={
        200: {
            "description": "Embeddings with segments generated successfully",
            "model": EmbedWithSegmentsResponse
        },
        422: {
            "description": "Invalid input validation"
        },
        429: {
            "description": "Too many concurrent requests"
        },
        503: {
            "description": "No model is currently loaded"
        },
        507: {
            "description": "Insufficient GPU memory"
        },
        500: {
            "description": "Embedding processing failed"
        }
    }
)
async def generate_embeddings_with_segments(request: EmbedRequest) -> EmbedWithSegmentsResponse:
    """
    Generate embeddings and return information about text segmentation.
    
    This endpoint is useful for understanding how long texts are processed:
    - Shows the actual text segments created for each input
    - Helps with debugging embedding quality for long documents
    - Provides insight into the segmentation strategy
    
    Args:
        request: Embedding request with texts and optional parameters
    
    Returns:
        EmbedWithSegmentsResponse: Embeddings, segments, shape, and processing time
    
    Raises:
        ModelNotLoadedException: If no model is currently loaded
        ValidationException: If input validation fails
        EmbeddingProcessingException: If embedding generation fails
        GPUMemoryException: If GPU memory is insufficient
        ConcurrencyLimitException: If service is at capacity
    """
    start_time = time.time()
    
    try:
        # Generate embeddings with segments using the manager
        result = await embedding_manager.embed_with_segments(
            texts=request.texts,
            show_progress=request.show_progress
        )
        
        # Unpack result based on whether it's single text or multiple texts
        if isinstance(request.texts, str):
            # Single text case
            embeddings, segments = result
            embeddings = embeddings.reshape(1, -1)  # Ensure 2D shape
            segments = [segments]  # Wrap in list for consistency
        else:
            # Multiple texts case
            embeddings, segments = result
        
        # Validate and convert numpy array to list for JSON serialization
        validate_embedding_shape(embeddings)
        embeddings_list = numpy_to_list(embeddings)
        processing_time = time.time() - start_time
        
        return EmbedWithSegmentsResponse(
            embeddings=embeddings_list,
            segments=segments,
            shape=list(embeddings.shape),
            processing_time=processing_time
        )
        
    except (ModelNotLoadedException, ValidationException, EmbeddingProcessingException, 
            GPUMemoryException, ConcurrencyLimitException):
        raise
    except ValueError as e:
        raise ValidationException(detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error generating embeddings with segments: {str(e)}"
        )


@router.post(
    "/quick",
    response_model=EmbedResponse,
    summary="Quick embedding generation",
    description="Generate embeddings with temporary model loading for one-off requests. Useful for testing different models.",
    responses={
        200: {
            "description": "Quick embeddings generated successfully",
            "model": EmbedResponse
        },
        400: {
            "description": "Invalid model name"
        },
        422: {
            "description": "Invalid input validation"
        },
        429: {
            "description": "Too many concurrent requests"
        },
        500: {
            "description": "Quick embedding processing failed"
        },
        507: {
            "description": "Insufficient GPU memory"
        }
    }
)
async def generate_quick_embeddings(request: QuickEmbedRequest) -> EmbedResponse:
    """
    Generate embeddings with temporary model loading.
    
    This endpoint creates a temporary EmbeddingEngine instance for the request,
    which is useful for:
    - Testing different models without changing the main service model
    - One-off embedding requests with specific model requirements
    - Smaller workloads that don't justify loading a persistent model
    
    Note: This endpoint has stricter limits on text length and batch size
    compared to the main embedding endpoint.
    
    Args:
        request: Quick embedding request with texts, model name, and options
    
    Returns:
        EmbedResponse: Generated embeddings with shape and processing time
    
    Raises:
        InvalidModelException: If the model name is not available
        ValidationException: If input validation fails
        EmbeddingProcessingException: If embedding generation fails
        GPUMemoryException: If GPU memory is insufficient
        ConcurrencyLimitException: If service is at capacity
    """
    start_time = time.time()
    
    async with embedding_manager.get_semaphore('embedding_requests'):
        try:
            # Validate model name against available models
            available_models = embedding_manager.available_models
            if request.model_name not in available_models:
                raise InvalidModelException(
                    model_name=request.model_name,
                    available_models=list(available_models.keys())
                )
            
            # Use convenience function for quick embedding
            embeddings = await quick_embed(
                request.texts,
                request.model_name,
                request.use_segmentation
            )
            
            # Validate and convert numpy array to list for JSON serialization
            validate_embedding_shape(embeddings)
            embeddings_list = numpy_to_list(embeddings)
            processing_time = time.time() - start_time
            
            return EmbedResponse(
                embeddings=embeddings_list,
                shape=list(embeddings.shape),
                processing_time=processing_time
            )
            
        except (InvalidModelException, ValidationException, EmbeddingProcessingException, 
                GPUMemoryException, ConcurrencyLimitException):
            raise
        except ValueError as e:
            raise ValidationException(detail=str(e))
        except Exception as e:
            # Handle GPU errors specifically
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg or "memory" in error_msg:
                raise GPUMemoryException(
                    detail=f"GPU error during quick embedding: {str(e)}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected error in quick embedding: {str(e)}"
                )


@router.get(
    "/stats",
    summary="Get embedding statistics",
    description="Get statistics about embedding operations and performance metrics.",
    responses={
        200: {
            "description": "Successfully retrieved embedding statistics"
        },
        429: {
            "description": "Too many concurrent requests"
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def get_embedding_stats():
    """
    Get statistics and performance metrics for embedding operations.
    
    Returns information about:
    - Current semaphore usage for embedding requests
    - Model performance characteristics
    - Resource utilization
    
    Returns:
        Dict containing embedding operation statistics
    
    Raises:
        ConcurrencyLimitException: If service is at capacity
    """
    async with embedding_manager.get_semaphore('health_checks'):
        try:
            stats = {
                'semaphore_status': embedding_manager.semaphore_status,
                'model_loaded': embedding_manager.is_model_loaded,
                'current_model': embedding_manager.current_model,
                'uptime_seconds': embedding_manager.uptime_seconds,
                'timestamp': time.time()
            }
            
            # Add model-specific stats if available
            if embedding_manager.is_model_loaded:
                try:
                    model_info = await embedding_manager.get_model_info()
                    stats['model_stats'] = {
                        'embedding_dim': model_info['embedding_dim'],
                        'max_length': model_info['max_length'],
                        'default_batch_size': model_info['default_batch_size'],
                        'device': model_info['device'],
                        'normalize_embeddings': model_info['normalize_embeddings']
                    }
                except ModelNotLoadedException:
                    stats['model_stats'] = None
            else:
                stats['model_stats'] = None
            
            # Add GPU memory info
            gpu_info = await embedding_manager.get_gpu_memory_info()
            stats['gpu_stats'] = gpu_info
            
            return stats
            
        except ConcurrencyLimitException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve embedding statistics: {str(e)}"
            )