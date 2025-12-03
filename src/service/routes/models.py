"""
Model management routes for the FastAPI Embedding Service.

Provides endpoints for managing embedding models including loading, unloading,
cache management, and model information retrieval.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from src.service.core.engine_manager import embedding_manager
from src.service.core.exceptions import (ConcurrencyLimitException,
                                         GPUMemoryException,
                                         InvalidModelException,
                                         ModelLoadingException,
                                         ModelNotLoadedException)
from src.service.schemas.requests import ModelLoadRequest
from src.service.schemas.responses import (AvailableModelsResponse,
                                           ErrorResponse, ModelInfoResponse,
                                           SuccessResponse)

# Create router
router = APIRouter(tags=["Model Management"])


@router.get(
    "/available",
    response_model=AvailableModelsResponse,
    summary="Get available models",
    description="Retrieve a list of all available embedding models that can be loaded.",
    responses={
        200: {
            "description": "Successfully retrieved available models",
            "model": AvailableModelsResponse
        },
        429: {
            "description": "Too many concurrent requests",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def get_available_models() -> AvailableModelsResponse:
    """
    Get a list of all available embedding models.

    Returns:
        AvailableModelsResponse: Dictionary of available models with their identifiers and paths

    Raises:
        ConcurrencyLimitException: If service is at capacity
    """
    async with embedding_manager.get_semaphore('health_checks'):
        try:
            available_models = embedding_manager.available_models
            current_model = embedding_manager.current_model

            return AvailableModelsResponse(
                available_models=available_models,
                current_model=current_model
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve available models: {str(e)}"
            )


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get current model information",
    description="Retrieve detailed information about the currently loaded embedding model.",
    responses={
        200: {
            "description": "Successfully retrieved model information",
            "model": ModelInfoResponse
        },
        429: {
            "description": "Too many concurrent requests",
            "model": ErrorResponse
        },
        503: {
            "description": "No model is currently loaded",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def get_model_info() -> ModelInfoResponse:
    """
    Get detailed information about the currently loaded model.

    Returns:
        ModelInfoResponse: Comprehensive model information including device, dimensions, etc.

    Raises:
        ModelNotLoadedException: If no model is currently loaded
        ConcurrencyLimitException: If service is at capacity
    """
    async with embedding_manager.get_semaphore('health_checks'):
        try:
            model_info = await embedding_manager.get_model_info()

            return ModelInfoResponse(
                model_path=model_info['model_path'],
                device=model_info['device'],
                embedding_dim=model_info['embedding_dim'],
                max_length=model_info['max_length'],
                default_batch_size=model_info['default_batch_size'],
                normalize_embeddings=model_info['normalize_embeddings'],
                gpu_available=model_info['gpu_available'],
                gpu_memory=model_info.get('gpu_memory')
            )

        except ModelNotLoadedException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve model information: {str(e)}"
            )


@router.post(
    "/load",
    response_model=SuccessResponse,
    summary="Load a specific model",
    description="Load a specific embedding model with optional configuration parameters.",
    responses={
        200: {
            "description": "Model loaded successfully",
            "model": SuccessResponse
        },
        400: {
            "description": "Invalid model name or parameters",
            "model": ErrorResponse
        },
        429: {
            "description": "Too many concurrent requests",
            "model": ErrorResponse
        },
        500: {
            "description": "Model loading failed",
            "model": ErrorResponse
        },
        507: {
            "description": "Insufficient GPU memory",
            "model": ErrorResponse
        }
    }
)
async def load_model(request: ModelLoadRequest) -> SuccessResponse:
    """
    Load a specific embedding model with optional configuration.

    Args:
        request: Model loading request with model name and optional parameters

    Returns:
        SuccessResponse: Success confirmation with processing time and model info

    Raises:
        InvalidModelException: If the model name is not available
        ModelLoadingException: If model loading fails
        GPUMemoryException: If insufficient GPU memory
        ConcurrencyLimitException: If service is at capacity
    """
    start_time = time.time()

    try:
        # Validate model name against available models
        available_models = embedding_manager.available_models
        if request.model_name not in available_models:
            raise InvalidModelException(
                model_name=request.model_name,
                available_models=list(available_models.keys())
            )

        # Load the model
        load_result = await embedding_manager.load_model(
            model_name=request.model_name,
            device=request.device,
            max_length=request.max_length,
            batch_size=request.batch_size,
            normalize_embeddings=request.normalize_embeddings
        )

        processing_time = time.time() - start_time

        return SuccessResponse(
            success=True,
            message=f"Model '{request.model_name}' loaded successfully in {load_result['load_time_seconds']:.2f}s",
            processing_time=processing_time
        )

    except (InvalidModelException, ModelLoadingException, GPUMemoryException, ConcurrencyLimitException):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error loading model: {str(e)}"
        )


@router.post(
    "/cache/clear",
    response_model=SuccessResponse,
    summary="Clear GPU cache",
    description="Clear GPU memory cache to free up VRAM. Useful when encountering out-of-memory errors.",
    responses={
        200: {
            "description": "Cache cleared successfully",
            "model": SuccessResponse
        },
        429: {
            "description": "Too many concurrent requests",
            "model": ErrorResponse
        },
        500: {
            "description": "Cache clearing failed",
            "model": ErrorResponse
        },
        507: {
            "description": "GPU operation failed",
            "model": ErrorResponse
        }
    }
)
async def clear_cache() -> SuccessResponse:
    """
    Clear GPU memory cache to free up VRAM.

    This endpoint clears both the model's internal cache and PyTorch's CUDA cache.
    Useful for recovering from out-of-memory situations or before loading large models.

    Returns:
        SuccessResponse: Success confirmation with processing time

    Raises:
        GPUMemoryException: If GPU cache clearing fails
        ConcurrencyLimitException: If service is at capacity
    """
    try:
        cache_result = await embedding_manager.clear_cache()

        return SuccessResponse(
            success=cache_result['success'],
            message=cache_result['message'],
            processing_time=cache_result['processing_time']
        )

    except (GPUMemoryException, ConcurrencyLimitException):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error clearing cache: {str(e)}"
        )


@router.get(
    "/status",
    summary="Get model status and semaphore information",
    description="Get detailed status about loaded models, semaphore usage, and resource utilization.",
    responses={
        200: {
            "description": "Successfully retrieved model status"
        },
        429: {
            "description": "Too many concurrent requests",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def get_model_status() -> Dict[str, Any]:
    """
    Get comprehensive model and resource status information.

    Returns detailed information about:
    - Currently loaded model
    - Semaphore usage and limits
    - GPU memory utilization
    - Service uptime

    Returns:
        Dict containing comprehensive status information

    Raises:
        ConcurrencyLimitException: If service is at capacity
    """
    async with embedding_manager.get_semaphore('health_checks'):
        try:
            status_info = {
                'model_loaded': embedding_manager.is_model_loaded,
                'current_model': embedding_manager.current_model,
                'uptime_seconds': embedding_manager.uptime_seconds,
                'semaphore_status': embedding_manager.semaphore_status,
                'timestamp': time.time()
            }

            # Add model info if available
            if embedding_manager.is_model_loaded:
                try:
                    model_info = await embedding_manager.get_model_info()
                    status_info['model_info'] = model_info
                except ModelNotLoadedException:
                    status_info['model_info'] = None
            else:
                status_info['model_info'] = None

            # Add GPU memory info
            gpu_info = await embedding_manager.get_gpu_memory_info()
            status_info['gpu_info'] = gpu_info

            # Add available models
            status_info['available_models'] = embedding_manager.available_models

            return status_info

        except ConcurrencyLimitException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve model status: {str(e)}"
            )
