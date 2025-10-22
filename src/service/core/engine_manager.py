"""
Singleton EmbeddingEngine manager for the FastAPI Embedding Service.

This module provides thread-safe management of EmbeddingEngine instances with
lazy loading, model switching, and resource cleanup capabilities.
"""

import asyncio
import threading
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import torch

from src.components.embedding_engine import EmbeddingEngine
from src.config import app_conf
from src.service.core.exceptions import (
    ModelNotLoadedException,
    ModelLoadingException,
    InvalidModelException,
    GPUMemoryException,
    ConcurrencyLimitException,
    handle_model_error,
    handle_gpu_error
)


class EmbeddingEngineManager:
    """
    Singleton manager for EmbeddingEngine instances with thread-safe operations
    and semaphore-based concurrency control.
    """
    
    _instance: Optional['EmbeddingEngineManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'EmbeddingEngineManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'EmbeddingEngineManager':
        """Get the singleton instance, creating it if necessary."""
        return cls()
    
    def __init__(self):
        """Initialize the manager (only called once due to singleton pattern)."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._engine: Optional[EmbeddingEngine] = None
        self._current_model: Optional[str] = None
        self._model_lock = asyncio.Lock()
        self._load_time: Optional[float] = None
        self._startup_time = time.time()
        
        # Semaphore configuration - hardcoded defaults (config system only supports 2-level keys)
        self._semaphore_config = {
            'gpu_operations': 2,
            'model_loading': 1,
            'embedding_requests': 4,
            'similarity_requests': 3,
            'health_checks': 10,
        }
        
        # Create semaphores
        self._semaphores = {
            'gpu_operations': asyncio.Semaphore(self._semaphore_config['gpu_operations']),
            'model_loading': asyncio.Semaphore(self._semaphore_config['model_loading']),
            'embedding_requests': asyncio.Semaphore(self._semaphore_config['embedding_requests']),
            'similarity_requests': asyncio.Semaphore(self._semaphore_config['similarity_requests']),
            'health_checks': asyncio.Semaphore(self._semaphore_config['health_checks']),
        }
        
        # Available models from EmbeddingEngine
        self._available_models = EmbeddingEngine.SMALL_MODELS.copy()
        
        # Default model from configuration (two-level key works!)
        self._default_model = app_conf.get('embedding.default_model') or 'all-mini'
        
        # Lazy loading flag - hardcoded default (no two-level service.auto_load config exists)
        self._auto_load_default = True
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._engine is not None
    
    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model identifier."""
        return self._current_model
    
    @property
    def available_models(self) -> Dict[str, str]:
        """Get the dictionary of available models."""
        return self._available_models.copy()
    
    @property
    def semaphore_status(self) -> Dict[str, Dict[str, int]]:
        """Get current semaphore status."""
        status = {}
        for name, semaphore in self._semaphores.items():
            status[name] = {
                'limit': self._semaphore_config[name],
                'available': semaphore._value,
                'in_use': self._semaphore_config[name] - semaphore._value
            }
        return status
    
    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._startup_time
    
    @asynccontextmanager
    async def get_semaphore(self, semaphore_name: str):
        """
        Async context manager for semaphore acquisition.
        
        Args:
            semaphore_name: Name of the semaphore to acquire
            
        Raises:
            ConcurrencyLimitException: If semaphore cannot be acquired
        """
        if semaphore_name not in self._semaphores:
            raise ValueError(f"Unknown semaphore: {semaphore_name}")
        
        semaphore = self._semaphores[semaphore_name]
        
        # Try to acquire with timeout
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=30.0)
            try:
                yield
            finally:
                semaphore.release()
        except asyncio.TimeoutError:
            limit = self._semaphore_config[semaphore_name]
            raise ConcurrencyLimitException(
                limit=limit,
                detail=f"Could not acquire {semaphore_name} semaphore within timeout (limit: {limit})"
            )
    
    async def ensure_model_loaded(self, model_name: Optional[str] = None) -> EmbeddingEngine:
        """
        Ensure a model is loaded, loading the default if necessary.
        
        Args:
            model_name: Specific model to ensure is loaded (uses current or default if None)
            
        Returns:
            The loaded EmbeddingEngine instance
            
        Raises:
            ModelNotLoadedException: If no model is loaded and auto-loading is disabled
            InvalidModelException: If the requested model is not available
        """
        async with self._model_lock:
            # If no specific model requested, use current or default
            if model_name is None:
                if self._engine is not None:
                    return self._engine
                elif self._auto_load_default:
                    model_name = self._default_model
                else:
                    raise ModelNotLoadedException()
            
            # If requested model is already loaded, return it
            if self._engine is not None and self._current_model == model_name:
                return self._engine
            
            # Load the requested model
            await self._load_model(model_name)
            return self._engine
    
    async def load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Load a specific model with optional parameters.
        
        Args:
            model_name: Model identifier to load
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            max_length: Maximum token length for input texts
            batch_size: Default batch size for processing
            normalize_embeddings: Whether to L2 normalize embeddings
            
        Returns:
            Dictionary with model information and load time
            
        Raises:
            InvalidModelException: If the model is not available
            ModelLoadingException: If model loading fails
        """
        async with self.get_semaphore('model_loading'):
            async with self._model_lock:
                start_time = time.time()
                
                # Validate model name
                if model_name not in self._available_models:
                    raise InvalidModelException(
                        model_name=model_name,
                        available_models=list(self._available_models.keys())
                    )
                
                try:
                    # Clear existing model if loaded
                    if self._engine is not None:
                        await self._cleanup_current_model()
                    
                    # Load new model
                    await self._load_model(
                        model_name=model_name,
                        device=device,
                        max_length=max_length,
                        batch_size=batch_size,
                        normalize_embeddings=normalize_embeddings
                    )
                    
                    load_time = time.time() - start_time
                    
                    return {
                        'model_name': model_name,
                        'model_path': self._available_models[model_name],
                        'load_time_seconds': load_time,
                        'model_info': self._engine.get_model_info()
                    }
                    
                except Exception as e:
                    raise handle_model_error(e, model_name, "loading")
    
    async def _load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None
    ):
        """Internal method to load a model."""
        try:
            # Create new engine instance
            self._engine = EmbeddingEngine(
                model_name=model_name,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings
            )
            
            self._current_model = model_name
            self._load_time = time.time()
            
        except Exception as e:
            self._engine = None
            self._current_model = None
            self._load_time = None
            
            # Handle specific error types
            if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                raise handle_gpu_error(e, f"loading model '{model_name}'")
            else:
                raise ModelLoadingException(
                    model_name=model_name,
                    original_error=e
                )
    
    async def _cleanup_current_model(self):
        """Clean up the currently loaded model."""
        if self._engine is not None:
            try:
                # Clear GPU cache if available
                self._engine.clear_cache()
            except Exception:
                pass  # Ignore cleanup errors
            
            self._engine = None
            self._current_model = None
            self._load_time = None
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
            
        Raises:
            ModelNotLoadedException: If no model is loaded
        """
        if self._engine is None:
            raise ModelNotLoadedException()
        
        info = await self._engine.get_model_info()
        info.update({
            'model_identifier': self._current_model,
            'load_time': self._load_time,
            'uptime_seconds': time.time() - self._load_time if self._load_time else None
        })
        
        return info
    
    async def clear_cache(self) -> Dict[str, Any]:
        """
        Clear GPU cache.
        
        Returns:
            Dictionary with cache clearing information
        """
        async with self.get_semaphore('gpu_operations'):
            start_time = time.time()
            
            try:
                if self._engine is not None:
                    self._engine.clear_cache()
                
                # Also clear PyTorch cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    'success': True,
                    'processing_time': time.time() - start_time,
                    'message': 'GPU cache cleared successfully'
                }
                
            except Exception as e:
                raise handle_gpu_error(e, "clearing cache")

    async def cleanup(self):
        """
        Cleanup resources held by the manager (called during application shutdown).
        This will attempt to clear the currently loaded model and free GPU memory.
        """
        try:
            await self._cleanup_current_model()
        except Exception:
            # Best-effort cleanup; don't raise during shutdown
            pass
    
    async def embed(self, *args, **kwargs):
        """Wrapper for embedding operations with semaphore control."""
        async with self.get_semaphore('embedding_requests'):
            engine = await self.ensure_model_loaded()
            
            try:
                # Call the async embed method directly
                return await engine.embed(*args, **kwargs)
            except Exception as e:
                if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                    raise handle_gpu_error(e, "embedding generation")
                else:
                    raise
    
    async def embed_with_segments(self, *args, **kwargs):
        """Wrapper for embedding with segments operations with semaphore control."""
        async with self.get_semaphore('embedding_requests'):
            engine = await self.ensure_model_loaded()
            
            try:
                # Call the async embed_with_segments method directly
                return await engine.embed_with_segments(*args, **kwargs)
            except Exception as e:
                if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                    raise handle_gpu_error(e, "embedding with segments generation")
                else:
                    raise
    
    async def similarity(self, *args, **kwargs):
        """Wrapper for similarity operations with semaphore control."""
        async with self.get_semaphore('similarity_requests'):
            engine = await self.ensure_model_loaded()
            
            try:
                # Call the async similarity method directly
                return await engine.similarity(*args, **kwargs)
            except Exception as e:
                if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                    raise handle_gpu_error(e, "similarity computation")
                else:
                    raise
            except Exception as e:
                if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                    raise handle_gpu_error(e, "similarity computation")
                else:
                    raise
    
    async def find_most_similar(self, *args, **kwargs):
        """Wrapper for similarity search operations with semaphore control."""
        async with self.get_semaphore('similarity_requests'):
            engine = await self.ensure_model_loaded()
            
            try:
                # Call the async find_most_similar method directly
                return await engine.find_most_similar(*args, **kwargs)
            except Exception as e:
                if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                    raise handle_gpu_error(e, "similarity search")
                else:
                    raise
    
    async def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {
                'gpu_available': False,
                'gpu_memory_used': None,
                'gpu_memory_total': None
            }
        
        try:
            device = torch.cuda.current_device()
            memory_used = torch.cuda.memory_allocated(device)
            memory_total = torch.cuda.get_device_properties(device).total_memory
            
            return {
                'gpu_available': True,
                'gpu_memory_used': memory_used,
                'gpu_memory_total': memory_total,
                'gpu_memory_used_gb': memory_used / (1024**3),
                'gpu_memory_total_gb': memory_total / (1024**3),
                'gpu_memory_percent': (memory_used / memory_total) * 100
            }
        except Exception:
            return {
                'gpu_available': True,
                'gpu_memory_used': None,
                'gpu_memory_total': None,
                'error': 'Failed to get GPU memory info'
            }
    
    async def shutdown(self):
        """Cleanup resources during shutdown."""
        async with self._model_lock:
            await self._cleanup_current_model()


# Global manager instance
# Global singleton instance
embedding_manager = EmbeddingEngineManager.get_instance()