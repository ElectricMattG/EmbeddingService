"""
Request validation middleware for the FastAPI Embedding Service.

This middleware provides comprehensive request validation including size limits,
rate limiting, and input validation.
"""

import asyncio
import time
import logging
from typing import Dict, Any
from collections import defaultdict
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.service.utils.validation import validate_request_size, ValidationError
from src.config import app_conf

logger = logging.getLogger(__name__)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating request sizes and implementing basic rate limiting.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.request_counts = defaultdict(list)  # IP -> list of timestamps
        # Read rate limit from configuration (fallback to 600 requests/min)
        try:
            configured = int(app_conf.get('service.rate_limit_per_minute') or 600)
            self.max_requests_per_minute = configured if configured > 0 else 600
        except Exception:
            self.max_requests_per_minute = 600
        self.cleanup_interval = 60  # Clean up old entries every minute
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request through validation pipeline.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        try:
            # Skip validation for health checks and docs
            if request.url.path in ["/health", "/health/ready", "/health/status", "/docs", "/openapi.json"]:
                return await call_next(request)
            
            # Basic rate limiting
            client_ip = self._get_client_ip(request)
            if not self._check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "type": "RateLimitExceeded",
                            "message": "Too many requests from this IP address",
                            "details": {
                                "limit_per_minute": self.max_requests_per_minute,
                                "retry_after_seconds": 60
                            }
                        }
                    }
                )
            
            # Validate request size for POST requests
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_request_size(request)
            
            # Process request
            response = await call_next(request)
            
            # Add processing time header
            processing_time = time.time() - start_time
            response.headers["X-Processing-Time"] = f"{processing_time:.4f}"
            
            return response
            
        except ValidationError as e:
            logger.warning(f"Request validation failed: {e.message}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": {
                        "type": "ValidationError",
                        "message": e.message,
                        "field": e.field,
                        "details": e.details
                    }
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in request validation: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "type": "InternalServerError",
                        "message": "Request processing failed",
                        "details": {"error": str(e)}
                    }
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for X-Real-IP header (nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client is within rate limits.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if within limits, False if rate limited
        """
        current_time = time.time()
        
        # Clean up old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Get request timestamps for this IP
        client_requests = self.request_counts[client_ip]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        client_requests[:] = [ts for ts in client_requests if ts > cutoff_time]
        
        # Check if under limit
        if len(client_requests) >= self.max_requests_per_minute:
            return False
        
        # Add current request
        client_requests.append(current_time)
        return True
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old rate limiting entries to prevent memory growth."""
        cutoff_time = current_time - 300  # Keep 5 minutes of history
        
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = [
                ts for ts in self.request_counts[ip] 
                if ts > cutoff_time
            ]
            # Remove empty entries
            if not self.request_counts[ip]:
                del self.request_counts[ip]
    
    async def _validate_request_size(self, request: Request):
        """
        Validate request body size.
        
        Args:
            request: HTTP request to validate
            
        Raises:
            ValidationError: If request is too large
        """
        # Get content length
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            max_size = app_conf.get('service.max_request_size') or 10485760
            
            if content_length > max_size:
                raise ValidationError(
                    f"Request body too large",
                    field="content_length",
                    details={
                        "max_size_mb": max_size / 1024 / 1024,
                        "received_size_mb": content_length / 1024 / 1024,
                        "max_size_bytes": max_size,
                        "received_size_bytes": content_length
                    }
                )