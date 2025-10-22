"""
Global error handling middleware for FastAPI EmbeddingService.

This middleware provides:
- Global exception handling for all custom exceptions
- Consistent error response formatting
- GPU memory error recovery
- Concurrency limit error handling
- Numpy serialization error handling
- Request context preservation in error responses
"""

import asyncio
import logging
import traceback
from typing import Dict, Any, Union
import numpy as np

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.exceptions import (
    ModelNotLoadedException,
    InvalidModelException, 
    EmbeddingProcessingException,
    GPUMemoryException,
    ConcurrencyLimitException,
    convert_to_http_exception
)

# Initialize logger
logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware that catches and processes all exceptions.
    
    Provides consistent error responses and logging for:
    - Custom business exceptions
    - HTTP exceptions
    - Validation errors
    - Numpy serialization errors
    - GPU memory errors
    - Unexpected exceptions
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process requests and handle any exceptions that occur.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in the chain
            
        Returns:
            Response object with appropriate error handling
        """
        try:
            # Process the request
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Handle the exception and return appropriate error response
            return await self._handle_exception(request, exc)
    
    async def _handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """
        Handle specific exception types and format error responses.
        
        Args:
            request: FastAPI request object
            exc: Exception that occurred
            
        Returns:
            JSONResponse with formatted error details
        """
        # Extract request context for logging
        request_context = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "query_params": dict(request.query_params),
        }
        
        # Handle different exception types
        if isinstance(exc, ModelNotLoadedException):
            return self._create_error_response(
                status_code=503,
                error_type="ModelNotLoaded",
                message=str(exc),
                details={"suggestion": "Load a model first using POST /models/load"},
                request_context=request_context
            )
        
        elif isinstance(exc, InvalidModelException):
            return self._create_error_response(
                status_code=400,
                error_type="InvalidModel",
                message=str(exc),
                details={"suggestion": "Check available models at GET /models/available"},
                request_context=request_context
            )
        
        elif isinstance(exc, EmbeddingProcessingException):
            logger.error(f"Embedding processing error: {exc}", extra=request_context)
            return self._create_error_response(
                status_code=500,
                error_type="EmbeddingProcessingError",
                message=str(exc),
                details={"suggestion": "Check input format and try again"},
                request_context=request_context
            )
        
        elif isinstance(exc, GPUMemoryException):
            logger.error(f"GPU memory error: {exc}", extra=request_context)
            return self._create_error_response(
                status_code=507,  # Insufficient Storage
                error_type="GPUMemoryError",
                message=str(exc),
                details={
                    "suggestion": "Try reducing batch size or clearing GPU cache",
                    "cache_clear_endpoint": "POST /models/cache/clear"
                },
                request_context=request_context
            )
        
        elif isinstance(exc, ConcurrencyLimitException):
            logger.warning(f"Concurrency limit reached: {exc}", extra=request_context)
            return self._create_error_response(
                status_code=429,  # Too Many Requests
                error_type="ConcurrencyLimitReached",
                message=str(exc),
                details={"suggestion": "Reduce concurrent requests or try again later"},
                request_context=request_context
            )
        
        elif isinstance(exc, HTTPException):
            # FastAPI HTTPException
            return self._create_error_response(
                status_code=exc.status_code,
                error_type="HTTPError",
                message=exc.detail,
                request_context=request_context
            )
        
        elif isinstance(exc, StarletteHTTPException):
            # Starlette HTTPException
            return self._create_error_response(
                status_code=exc.status_code,
                error_type="HTTPError", 
                message=exc.detail,
                request_context=request_context
            )
        
        elif isinstance(exc, RequestValidationError):
            # Pydantic validation error
            return self._handle_validation_error(exc, request_context)
        
        elif isinstance(exc, ValidationError):
            # General validation error
            return self._handle_validation_error(exc, request_context)
        
        elif isinstance(exc, (TypeError, ValueError)) and "JSON" in str(exc):
            # Likely numpy serialization error
            logger.error(f"JSON serialization error: {exc}", extra=request_context)
            return self._create_error_response(
                status_code=500,
                error_type="SerializationError",
                message="Failed to serialize response data",
                details={
                    "original_error": str(exc),
                    "suggestion": "Contact support if this persists"
                },
                request_context=request_context
            )
        
        elif isinstance(exc, asyncio.TimeoutError):
            # Request timeout
            logger.error(f"Request timeout: {exc}", extra=request_context)
            return self._create_error_response(
                status_code=408,  # Request Timeout
                error_type="RequestTimeout",
                message="Request processing timed out",
                details={"suggestion": "Try reducing batch size or try again later"},
                request_context=request_context
            )
        
        else:
            # Unexpected error - log full traceback
            logger.error(
                f"Unexpected error in {request.method} {request.url.path}: {exc}",
                exc_info=True,
                extra=request_context
            )
            return self._create_error_response(
                status_code=500,
                error_type="InternalServerError",
                message="An unexpected error occurred",
                details={
                    "error_id": id(exc),  # Simple error ID for tracking
                    "suggestion": "Please try again or contact support"
                },
                request_context=request_context
            )
    
    def _handle_validation_error(
        self, 
        exc: Union[RequestValidationError, ValidationError], 
        request_context: Dict[str, Any]
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors with detailed field information.
        
        Args:
            exc: Validation error from Pydantic
            request_context: Request context for logging
            
        Returns:
            JSONResponse with validation error details
        """
        error_details = []
        
        if hasattr(exc, 'errors'):
            for error in exc.errors():
                error_details.append({
                    "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
                    "message": error.get("msg", "Validation error"),
                    "type": error.get("type", "unknown"),
                    "input": error.get("input", "not provided")
                })
        
        logger.warning(f"Validation error: {exc}", extra=request_context)
        
        return self._create_error_response(
            status_code=422,  # Unprocessable Entity
            error_type="ValidationError",
            message="Request validation failed",
            details={
                "errors": error_details,
                "suggestion": "Check the request format and field values"
            },
            request_context=request_context
        )
    
    def _create_error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
        details: Dict[str, Any] = None,
        request_context: Dict[str, Any] = None
    ) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            status_code: HTTP status code
            error_type: Type of error for client identification
            message: Human-readable error message
            details: Additional error details and suggestions
            request_context: Request context information
            
        Returns:
            JSONResponse with standardized error format
        """
        error_response = {
            "error": {
                "type": error_type,
                "message": message,
                "status_code": status_code,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        
        if details:
            error_response["error"]["details"] = details
        
        # Add request context for debugging (optional in production)
        if request_context and logger.isEnabledFor(logging.DEBUG):
            error_response["error"]["request_context"] = request_context
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )


def add_exception_handlers(app: FastAPI) -> None:
    """
    Add global exception handlers to FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(ModelNotLoadedException)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "ModelNotLoaded",
                    "message": str(exc),
                    "details": {"suggestion": "Load a model first using POST /models/load"}
                }
            }
        )
    
    @app.exception_handler(InvalidModelException)
    async def invalid_model_handler(request: Request, exc: InvalidModelException):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "InvalidModel",
                    "message": str(exc),
                    "details": {"suggestion": "Check available models at GET /models/available"}
                }
            }
        )
    
    @app.exception_handler(EmbeddingProcessingException)
    async def embedding_processing_handler(request: Request, exc: EmbeddingProcessingException):
        logger.error(f"Embedding processing error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "EmbeddingProcessingError",
                    "message": str(exc),
                    "details": {"suggestion": "Check input format and try again"}
                }
            }
        )
    
    @app.exception_handler(GPUMemoryException)
    async def gpu_memory_handler(request: Request, exc: GPUMemoryException):
        logger.error(f"GPU memory error: {exc}")
        return JSONResponse(
            status_code=507,
            content={
                "error": {
                    "type": "GPUMemoryError",
                    "message": str(exc),
                    "details": {
                        "suggestion": "Try reducing batch size or clearing GPU cache",
                        "cache_clear_endpoint": "POST /models/cache/clear"
                    }
                }
            }
        )
    
    @app.exception_handler(ConcurrencyLimitException)
    async def concurrency_limit_handler(request: Request, exc: ConcurrencyLimitException):
        logger.warning(f"Concurrency limit reached: {exc}")
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "type": "ConcurrencyLimitReached",
                    "message": str(exc),
                    "details": {"suggestion": "Reduce concurrent requests or try again later"}
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown")
            })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": {
                        "errors": error_details,
                        "suggestion": "Check the request format and field values"
                    }
                }
            }
        )
    
    logger.info("Global exception handlers registered successfully")