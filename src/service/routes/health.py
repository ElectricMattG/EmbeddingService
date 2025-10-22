"""
Health and status routes for the FastAPI Embedding Service.

Provides endpoints for monitoring service health, readiness, and detailed
status information including resource utilization.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from src.service.core.engine_manager import embedding_manager
from src.service.schemas.responses import HealthResponse, StatusResponse, ModelInfoResponse
from src.service.core.exceptions import (
    ModelNotLoadedException,
    ConcurrencyLimitException
)

# Create router
router = APIRouter(tags=["Health & Status"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Simple health check endpoint that returns service status. Lightweight endpoint for load balancers.",
    responses={
        200: {
            "description": "Service is healthy",
            "model": HealthResponse
        },
        503: {
            "description": "Service is unhealthy"
        }
    }
)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    This is a lightweight endpoint designed for:
    - Load balancer health checks
    - Basic service availability monitoring
    - Quick status verification
    
    Returns:
        HealthResponse: Basic health status and timestamp
    """
    try:
        # Basic health check - just verify the service is responding
        timestamp = datetime.now(timezone.utc).isoformat()
        
        return HealthResponse(
            status="healthy",
            timestamp=timestamp
        )
        
    except Exception as e:
        # If we can't even return a basic response, service is unhealthy
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service health check failed: {str(e)}"
        )


@router.get(
    "/ready",
    summary="Readiness check",
    description="Readiness check that verifies the service is ready to handle requests including model availability.",
    responses={
        200: {
            "description": "Service is ready"
        },
        503: {
            "description": "Service is not ready"
        }
    }
)
async def readiness_check():
    """
    Readiness check endpoint.
    
    This endpoint verifies that the service is ready to handle requests by checking:
    - Basic service functionality
    - Model loading status (if auto-load is enabled)
    - Critical resource availability
    
    Returns:
        Dict with readiness status and details
    
    Raises:
        HTTPException: If service is not ready
    """
    try:
        # Use health checks semaphore with minimal blocking
        async with embedding_manager.get_semaphore('health_checks'):
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Check if auto-load is enabled and we should have a model
            auto_load_default = getattr(embedding_manager, '_auto_load_default', True)
            
            readiness_status = {
                'status': 'ready',
                'timestamp': timestamp,
                'model_loaded': embedding_manager.is_model_loaded,
                'auto_load_enabled': auto_load_default,
                'uptime_seconds': embedding_manager.uptime_seconds
            }
            
            # If auto-load is enabled but no model is loaded, try to ensure default model
            if auto_load_default and not embedding_manager.is_model_loaded:
                try:
                    await embedding_manager.ensure_model_loaded()
                    readiness_status['model_loaded'] = True
                    readiness_status['message'] = 'Default model loaded during readiness check'
                except Exception as e:
                    readiness_status['status'] = 'not_ready'
                    readiness_status['message'] = f'Failed to load default model: {str(e)}'
                    
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=readiness_status
                    )
            
            return readiness_status
            
    except ConcurrencyLimitException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready: too many concurrent requests"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Detailed service status",
    description="Comprehensive service status including model info, resource utilization, and performance metrics.",
    responses={
        200: {
            "description": "Successfully retrieved service status",
            "model": StatusResponse
        },
        429: {
            "description": "Too many concurrent requests"
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def get_service_status() -> StatusResponse:
    """
    Get comprehensive service status information.
    
    This endpoint provides detailed information about:
    - Service health and uptime
    - Currently loaded model and its configuration
    - GPU memory utilization
    - Semaphore usage and concurrency status
    - Performance metrics
    
    Returns:
        StatusResponse: Comprehensive service status information
    
    Raises:
        ConcurrencyLimitException: If service is at capacity
    """
    async with embedding_manager.get_semaphore('health_checks'):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            uptime_seconds = embedding_manager.uptime_seconds
            
            # Get model information if available
            model_info = None
            if embedding_manager.is_model_loaded:
                try:
                    raw_model_info = await embedding_manager.get_model_info()
                    model_info = ModelInfoResponse(
                        model_path=raw_model_info['model_path'],
                        device=raw_model_info['device'],
                        embedding_dim=raw_model_info['embedding_dim'],
                        max_length=raw_model_info['max_length'],
                        default_batch_size=raw_model_info['default_batch_size'],
                        normalize_embeddings=raw_model_info['normalize_embeddings'],
                        gpu_available=raw_model_info['gpu_available'],
                        gpu_memory=raw_model_info.get('gpu_memory')
                    )
                except ModelNotLoadedException:
                    model_info = None
            
            # Get GPU memory information
            gpu_info = await embedding_manager.get_gpu_memory_info()
            
            return StatusResponse(
                status="healthy",
                model_loaded=embedding_manager.is_model_loaded,
                model_info=model_info,
                gpu_available=gpu_info['gpu_available'],
                gpu_memory_used=gpu_info.get('gpu_memory_used'),
                gpu_memory_total=gpu_info.get('gpu_memory_total'),
                uptime_seconds=uptime_seconds,
                timestamp=timestamp
            )
            
        except ConcurrencyLimitException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve service status: {str(e)}"
            )


@router.get(
    "/metrics",
    summary="Performance metrics",
    description="Get detailed performance metrics including semaphore usage, timing statistics, and resource utilization.",
    responses={
        200: {
            "description": "Successfully retrieved performance metrics"
        },
        429: {
            "description": "Too many concurrent requests"
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get detailed performance metrics and resource utilization.
    
    This endpoint provides comprehensive metrics including:
    - Semaphore usage across all operation types
    - GPU memory utilization and efficiency
    - Service uptime and availability
    - Model performance characteristics
    
    Returns:
        Dict containing detailed performance metrics
    
    Raises:
        ConcurrencyLimitException: If service is at capacity
    """
    async with embedding_manager.get_semaphore('health_checks'):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Get comprehensive metrics
            metrics = {
                'timestamp': timestamp,
                'uptime_seconds': embedding_manager.uptime_seconds,
                'service_status': 'healthy',
                'model_status': {
                    'loaded': embedding_manager.is_model_loaded,
                    'current_model': embedding_manager.current_model,
                },
                'semaphore_metrics': embedding_manager.semaphore_status,
                'available_models': embedding_manager.available_models
            }
            
            # Add model-specific metrics if available
            if embedding_manager.is_model_loaded:
                try:
                    model_info = await embedding_manager.get_model_info()
                    metrics['model_metrics'] = {
                        'model_path': model_info['model_path'],
                        'device': model_info['device'],
                        'embedding_dim': model_info['embedding_dim'],
                        'max_length': model_info['max_length'],
                        'default_batch_size': model_info['default_batch_size'],
                        'normalize_embeddings': model_info['normalize_embeddings'],
                        'load_time': model_info.get('load_time'),
                        'uptime_seconds': model_info.get('uptime_seconds')
                    }
                except ModelNotLoadedException:
                    metrics['model_metrics'] = None
            else:
                metrics['model_metrics'] = None
            
            # Add GPU metrics
            gpu_info = await embedding_manager.get_gpu_memory_info()
            metrics['gpu_metrics'] = gpu_info
            
            # Add efficiency metrics if GPU info is available
            if gpu_info['gpu_available'] and gpu_info.get('gpu_memory_used') and gpu_info.get('gpu_memory_total'):
                used_gb = gpu_info.get('gpu_memory_used_gb', 0)
                total_gb = gpu_info.get('gpu_memory_total_gb', 1)
                metrics['gpu_metrics']['efficiency'] = {
                    'memory_utilization_percent': gpu_info.get('gpu_memory_percent', 0),
                    'memory_available_gb': total_gb - used_gb,
                    'memory_efficiency_score': min(used_gb / total_gb * 100, 100) if total_gb > 0 else 0
                }
            
            # Add concurrency metrics
            semaphore_status = embedding_manager.semaphore_status
            total_capacity = sum(sem['limit'] for sem in semaphore_status.values())
            total_in_use = sum(sem['in_use'] for sem in semaphore_status.values())
            
            metrics['concurrency_metrics'] = {
                'total_capacity': total_capacity,
                'total_in_use': total_in_use,
                'total_available': total_capacity - total_in_use,
                'utilization_percent': (total_in_use / total_capacity * 100) if total_capacity > 0 else 0,
                'semaphore_breakdown': semaphore_status
            }
            
            return metrics
            
        except ConcurrencyLimitException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve performance metrics: {str(e)}"
            )


@router.get(
    "/live",
    summary="Liveness probe",
    description="Kubernetes-compatible liveness probe endpoint. Returns 200 if service is alive.",
    responses={
        200: {
            "description": "Service is alive"
        },
        500: {
            "description": "Service is not responding"
        }
    }
)
async def liveness_probe():
    """
    Liveness probe endpoint for Kubernetes health checks.
    
    This endpoint is designed for Kubernetes liveness probes and should
    only fail if the service process is completely unresponsive.
    
    Returns:
        Dict with minimal alive status
    """
    try:
        return {
            'status': 'alive',
            'timestamp': time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Liveness probe failed: {str(e)}"
        )