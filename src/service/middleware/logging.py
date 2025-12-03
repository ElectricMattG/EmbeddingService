"""
Request/response logging middleware for FastAPI EmbeddingService.

This middleware provides:
- Request/response logging with processing time tracking
- Performance metrics collection
- Error logging with request context
- Semaphore usage metrics
- Integration with existing Base class logging system
"""

import asyncio
import json
import logging
import os
import time
import uuid
from functools import wraps
from typing import Any, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ..core.dependencies import get_semaphore_limits

# Initialize logger
logger = logging.getLogger(__name__)

# Performance metrics storage (in production, use proper metrics system)
_metrics = {
    "request_count": 0,
    "total_processing_time": 0.0,
    "error_count": 0,
    "endpoint_metrics": {},
    "semaphore_usage": {}
}


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging and metrics.

    Tracks:
    - Request/response details
    - Processing times
    - Error rates
    - Semaphore usage
    - Performance metrics
    """

    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        """
        Initialize logging middleware.

        Args:
            app: FastAPI application instance
            log_requests: Whether to log incoming requests
            log_responses: Whether to log outgoing responses
        """
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses

    async def dispatch(self, request: Request, call_next):
        """
        Process request and log details with timing.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in the chain

        Returns:
            Response with logging and metrics collection
        """
        # Start timing
        start_time = time.time()

        # Extract request context
        request_context = self._extract_request_context(request)

        # Log incoming request
        if self.log_requests:
            await self._log_request(request, request_context)

        # Update request metrics
        self._update_request_metrics(request_context)

        try:
            # Process the request
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Log successful response
            if self.log_responses:
                await self._log_response(request, response, processing_time, request_context)

            # Update success metrics
            self._update_success_metrics(request_context, processing_time)

            return response

        except Exception as exc:
            # Calculate processing time for errors
            processing_time = time.time() - start_time

            # Log error
            await self._log_error(request, exc, processing_time, request_context)

            # Update error metrics
            self._update_error_metrics(request_context, processing_time)

            # Re-raise the exception
            raise exc

    def _extract_request_context(self, request: Request) -> Dict[str, Any]:
        """
        Extract useful context from the request.

        Args:
            request: FastAPI request object

        Returns:
            Dictionary with request context
        """
        return {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "timestamp": time.time()
        }

    async def _log_request(self, request: Request, context: Dict[str, Any]) -> None:
        """
        Log incoming request details.

        Args:
            request: FastAPI request object
            context: Request context
        """
        # Basic request logging
        logger.info(
            f"Incoming {context['method']} {context['path']} "
            f"from {context['client_ip']}",
            extra={
                "request_context": context,
                "event_type": "request"
            }
        )

        # Log request body for POST/PUT requests (truncated for large bodies)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body (this consumes the stream, so we need to be careful)
                body = await self._get_request_body(request)
                if body:
                    # Truncate large bodies
                    body_str = body[:1000] + "..." if len(body) > 1000 else body
                    logger.debug(
                        f"Request body: {body_str}",
                        extra={
                            "request_context": context,
                            "event_type": "request_body"
                        }
                    )
            except Exception as e:
                logger.debug(f"Could not read request body: {e}")

    async def _get_request_body(self, request: Request) -> str:
        """
        Safely read request body as string.

        Args:
            request: FastAPI request object

        Returns:
            Request body as string, or empty string if not available
        """
        try:
            if hasattr(request, '_body'):
                # Body already read
                return request._body.decode('utf-8')
            else:
                # Read body (this consumes the stream)
                body = await request.body()
                return body.decode('utf-8')
        except Exception:
            return ""

    async def _log_response(
        self,
        request: Request,
        response: Response,
        processing_time: float,
        context: Dict[str, Any]
    ) -> None:
        """
        Log response details.

        Args:
            request: FastAPI request object
            response: Response object
            processing_time: Time taken to process request
            context: Request context
        """
        # Attach lightweight tracing headers
        try:
            # X-Request-Id: reuse incoming header if present, else generate one
            req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
            response.headers.setdefault("X-Request-Id", req_id)
            # X-Replica: help correlate which pod/replica served the request
            replica = os.getenv("HOSTNAME") or "unknown"
            response.headers.setdefault("X-Replica", replica)
        except Exception:
            # Never fail response due to header decoration
            pass

        logger.info(
            f"Response {response.status_code} for {context['method']} {context['path']} "
            f"in {processing_time:.3f}s",
            extra={
                "request_context": context,
                "response_status": response.status_code,
                "processing_time": processing_time,
                "event_type": "response"
            }
        )

        # Log response headers if debug logging is enabled
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Response headers: {dict(response.headers)}",
                extra={
                    "request_context": context,
                    "event_type": "response_headers"
                }
            )

    async def _log_error(
        self,
        request: Request,
        exc: Exception,
        processing_time: float,
        context: Dict[str, Any]
    ) -> None:
        """
        Log error details with full context.

        Args:
            request: FastAPI request object
            exc: Exception that occurred
            processing_time: Time taken before error
            context: Request context
        """
        logger.error(
            f"Error in {context['method']} {context['path']}: {str(exc)} "
            f"after {processing_time:.3f}s",
            exc_info=True,
            extra={
                "request_context": context,
                "processing_time": processing_time,
                "error_type": type(exc).__name__,
                "event_type": "error"
            }
        )

    def _update_request_metrics(self, context: Dict[str, Any]) -> None:
        """
        Update request count metrics.

        Args:
            context: Request context
        """
        global _metrics

        _metrics["request_count"] += 1

        # Track per-endpoint metrics
        endpoint = f"{context['method']} {context['path']}"
        if endpoint not in _metrics["endpoint_metrics"]:
            _metrics["endpoint_metrics"][endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "error_count": 0
            }

        _metrics["endpoint_metrics"][endpoint]["count"] += 1

    def _update_success_metrics(self, context: Dict[str, Any], processing_time: float) -> None:
        """
        Update success metrics.

        Args:
            context: Request context
            processing_time: Time taken to process request
        """
        global _metrics

        _metrics["total_processing_time"] += processing_time

        # Update per-endpoint metrics
        endpoint = f"{context['method']} {context['path']}"
        if endpoint in _metrics["endpoint_metrics"]:
            _metrics["endpoint_metrics"][endpoint]["total_time"] += processing_time

    def _update_error_metrics(self, context: Dict[str, Any], processing_time: float) -> None:
        """
        Update error metrics.

        Args:
            context: Request context
            processing_time: Time taken before error
        """
        global _metrics

        _metrics["error_count"] += 1
        _metrics["total_processing_time"] += processing_time

        # Update per-endpoint error metrics
        endpoint = f"{context['method']} {context['path']}"
        if endpoint in _metrics["endpoint_metrics"]:
            _metrics["endpoint_metrics"][endpoint]["error_count"] += 1
            _metrics["endpoint_metrics"][endpoint]["total_time"] += processing_time


def log_semaphore_usage(operation_type: str):
    """
    Decorator to log semaphore acquisition and usage.

    Args:
        operation_type: Type of operation for semaphore tracking

    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            # Log semaphore acquisition attempt
            logger.debug(f"Attempting to acquire semaphore for {operation_type}")

            try:
                # Execute the function (which should acquire semaphore)
                result = await func(*args, **kwargs)

                # Log successful completion
                processing_time = time.time() - start_time
                logger.debug(
                    f"Completed {operation_type} operation in {processing_time:.3f}s",
                    extra={
                        "operation_type": operation_type,
                        "processing_time": processing_time,
                        "event_type": "semaphore_success"
                    }
                )

                # Update semaphore usage metrics
                _update_semaphore_metrics(operation_type, processing_time, success=True)

                return result

            except Exception as exc:
                # Log semaphore operation failure
                processing_time = time.time() - start_time
                logger.warning(
                    f"Failed {operation_type} operation after {processing_time:.3f}s: {exc}",
                    extra={
                        "operation_type": operation_type,
                        "processing_time": processing_time,
                        "error_type": type(exc).__name__,
                        "event_type": "semaphore_error"
                    }
                )

                # Update semaphore usage metrics
                _update_semaphore_metrics(operation_type, processing_time, success=False)

                raise exc

        return wrapper
    return decorator


def _update_semaphore_metrics(operation_type: str, processing_time: float, success: bool) -> None:
    """
    Update semaphore usage metrics.

    Args:
        operation_type: Type of operation
        processing_time: Time taken for operation
        success: Whether operation succeeded
    """
    global _metrics

    if operation_type not in _metrics["semaphore_usage"]:
        _metrics["semaphore_usage"][operation_type] = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }

    metrics = _metrics["semaphore_usage"][operation_type]
    metrics["total_operations"] += 1
    metrics["total_time"] += processing_time

    if success:
        metrics["successful_operations"] += 1
    else:
        metrics["failed_operations"] += 1

    # Update average time
    metrics["average_time"] = metrics["total_time"] / metrics["total_operations"]


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get current performance metrics.

    Returns:
        Dictionary with performance metrics
    """
    global _metrics

    # Calculate average response time
    avg_response_time = (
        _metrics["total_processing_time"] / _metrics["request_count"]
        if _metrics["request_count"] > 0 else 0.0
    )

    # Calculate error rate
    error_rate = (
        _metrics["error_count"] / _metrics["request_count"]
        if _metrics["request_count"] > 0 else 0.0
    )

    # Get semaphore limits for context
    semaphore_limits = get_semaphore_limits()

    return {
        "overall": {
            "total_requests": _metrics["request_count"],
            "total_errors": _metrics["error_count"],
            "error_rate": error_rate,
            "average_response_time": avg_response_time,
            "total_processing_time": _metrics["total_processing_time"]
        },
        "endpoints": _metrics["endpoint_metrics"],
        "semaphore_usage": _metrics["semaphore_usage"],
        "semaphore_limits": semaphore_limits,
        "timestamp": time.time()
    }


def reset_metrics() -> None:
    """Reset all performance metrics."""
    global _metrics
    _metrics = {
        "request_count": 0,
        "total_processing_time": 0.0,
        "error_count": 0,
        "endpoint_metrics": {},
        "semaphore_usage": {}
    }
    logger.info("Performance metrics reset")


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the service.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)

    # Configure formatter for structured logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handler to root logger if not already present
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    logger.info(f"Logging configured with level: {log_level}")
