"""
FastAPI Embedding Service

A high-performance, GPU-accelerated text embedding service built on FastAPI
that provides RESTful endpoints for generating embeddings using small BERT models.

Features:
- Multiple BERT model support
- GPU acceleration with CUDA
- Automatic text segmentation for long documents
- Similarity computation and search
- Async endpoints with semaphore-based concurrency control
- Comprehensive monitoring and health checks
"""

from .main import app, create_app

__version__ = "0.0.1"
__author__ = "Matt Gerhold"
__all__ = ["app", "create_app"]
