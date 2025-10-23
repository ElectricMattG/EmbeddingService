"""
Main FastAPI application for EmbeddingService.

This module provides:
- FastAPI application initialization and configuration
- Router registration for all API endpoints
- Middleware setup (CORS, error handling, logging)
- Startup/shutdown event handlers
- Global semaphore initialization
- Integration with existing configuration system
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.middleware.gzip import GZipMiddleware

from .core.dependencies import initialize_semaphores
from .middleware.logging import get_performance_metrics
from .core.engine_manager import EmbeddingEngineManager
from .middleware.error_handler import ErrorHandlerMiddleware, add_exception_handlers
from .middleware.logging import LoggingMiddleware, configure_logging
from .middleware.validation import RequestValidationMiddleware
from .routes import models, embeddings, similarity, health

# Initialize logger
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for application lifespan events.
    
    Handles:
    - Startup: Initialize semaphores, configure logging
    - Shutdown: Cleanup resources, model unloading
    """
    # Startup
    logger.info("Starting EmbeddingService...")
    
    try:
        # Initialize semaphores for concurrency control
        initialize_semaphores()
        logger.info("Semaphores initialized successfully")
        
        # Initialize embedding manager (singleton)
        manager = EmbeddingEngineManager.get_instance()
        logger.info("EmbeddingEngine manager initialized")
        
        # Optional: Pre-load a default model
        try:
            from src.config import app_conf
            default_model = app_conf.get("embedding.default_model")
            if default_model:
                logger.info(f"Pre-loading default model: {default_model}")
                await manager.load_model(default_model)
                logger.info("Default model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not pre-load default model: {e}")
        
        logger.info("EmbeddingService startup completed successfully")
        
        yield  # Application runs here
        
    finally:
        # Shutdown
        logger.info("Shutting down EmbeddingService...")
        
        try:
            # Cleanup embedding manager
            manager = EmbeddingEngineManager.get_instance()
            await manager.cleanup()
            logger.info("EmbeddingEngine manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("EmbeddingService shutdown completed")


def create_app(
    title: str = "EmbeddingService API",
    description: str = "FastAPI service for text embedding and similarity operations",
    version: str = "1.0.0",
    log_level: str = "INFO",
    enable_cors: bool = True,
    enable_metrics: bool = True
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        title: API title for documentation
        description: API description for documentation
        version: API version
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_cors: Whether to enable CORS middleware
        enable_metrics: Whether to enable performance metrics
        
    Returns:
        Configured FastAPI application
    """
    # Configure logging first
    configure_logging(log_level)
    
    # Create FastAPI app with lifespan management
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware (order matters - last added is executed first)
    
    # 1. CORS middleware (if enabled)
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS middleware enabled")
    
    # 2. Request validation middleware
    app.add_middleware(RequestValidationMiddleware)
    logger.info("Request validation middleware enabled")
    
    # 3. Enable gzip compression for large JSON responses to reduce network load
    #    This can help lower transient upstream errors under burst traffic and
    #    speeds up large payloads like segment embeddings.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    logger.info("GZip middleware enabled (minimum_size=1024 bytes)")

    # 4. Logging middleware
    app.add_middleware(LoggingMiddleware, log_requests=True, log_responses=True)
    logger.info("Logging middleware enabled")
    
    # 5. Error handling middleware
    app.add_middleware(ErrorHandlerMiddleware)
    logger.info("Error handling middleware enabled")
    
    # Add exception handlers
    add_exception_handlers(app)
    logger.info("Exception handlers registered")
    
    # Register route modules
    app.include_router(
        models.router,
        prefix="/models",
        tags=["Model Management"],
    )
    
    app.include_router(
        embeddings.router,
        prefix="/embeddings",
        tags=["Embedding Operations"],
    )
    
    app.include_router(
        similarity.router,
        prefix="/similarity",
        tags=["Similarity Operations"],
    )
    
    app.include_router(
        health.router,
        prefix="/health",
        tags=["Health & Status"],
    )
    
    logger.info("All route modules registered")
    
    # Add metrics endpoint if enabled
    if enable_metrics:
        
        @app.get("/metrics", tags=["Monitoring"])
        async def get_metrics():
            """
            Get service performance metrics.
            
            Returns detailed metrics including:
            - Request counts and error rates
            - Average response times
            - Per-endpoint statistics
            - Semaphore usage statistics
            """
            return get_performance_metrics()
        
        logger.info("Metrics endpoint enabled at /metrics")
    
    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """
        Root endpoint with service information.
        
        Returns basic service information and available endpoints.
        """
        return {
            "service": "EmbeddingService",
            "version": version,
            "status": "running",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "models": "/models",
                "embeddings": "/embeddings",
                "similarity": "/similarity"
            },
            "features": [
                "Text embedding generation",
                "Similarity computation",
                "Model management",
                "Async processing with semaphore control",
                "Comprehensive error handling",
                "Performance monitoring"
            ]
        }
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=title,
            version=version,
            description=description,
            routes=app.routes,
        )
        
        # Add custom information
        openapi_schema["info"]["x-service-features"] = [
            "Async embedding operations with GPU support",
            "Semaphore-based concurrency control",
            "Comprehensive model management",
            "Real-time health monitoring",
            "Structured error handling"
        ]
        
        openapi_schema["info"]["contact"] = {
            "name": "EmbeddingService API Support",
            "url": "https://github.com/your-org/embedding-service",
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    logger.info(f"FastAPI application created successfully: {title} v{version}")
    return app


# Create the default application instance
app = create_app()


# Development server configuration
if __name__ == "__main__":
    import uvicorn
    
    # Development server settings
    uvicorn.run(
        "src.service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="debug",
        access_log=True
    )