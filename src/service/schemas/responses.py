"""
Pydantic response models for the FastAPI Embedding Service.

These models define the structure for outgoing responses from all service endpoints.
"""

from typing import List, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    
    embeddings: List[List[float]] = Field(
        ..., 
        description="Generated embeddings as 2D list of floats",
        example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    shape: Tuple[int, int] = Field(
        ..., 
        description="Shape of the embeddings matrix (n_texts, embedding_dim)",
        example=[2, 384]
    )
    processing_time: float = Field(
        ..., 
        description="Time taken to process the request in seconds",
        example=0.125
    )
    segment_embeddings: Optional[Union[List[List[float]], List[List[List[float]]]]] = Field(
        None,
        description=(
            "Optional per-segment embeddings. For a single-text request: List[List[float]]. "
            "For multiple texts: List[List[List[float]]]. Present only when requested and when segmentation is enabled."
        )
    )
    
    class Config:
        schema_extra = {
            "example": {
                "embeddings": [[0.1, -0.2, 0.3], [0.4, -0.1, 0.2]],
                "shape": [2, 3],
                "processing_time": 0.125
            }
        }


class EmbedWithSegmentsResponse(BaseModel):
    """Response model for embedding generation with segmentation info."""
    
    embeddings: List[List[float]] = Field(
        ..., 
        description="Generated embeddings as 2D list of floats",
        example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    segments: Union[List[str], List[List[str]]] = Field(
        ..., 
        description="Text segments used for embedding generation",
        example=[["First segment", "Second segment"], ["Another text segment"]]
    )
    shape: Tuple[int, int] = Field(
        ..., 
        description="Shape of the embeddings matrix (n_texts, embedding_dim)",
        example=[2, 384]
    )
    processing_time: float = Field(
        ..., 
        description="Time taken to process the request in seconds",
        example=0.150
    )
    segment_embeddings: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., 
        description=(
            "Per-segment embeddings. For a single-text request: List[List[float]]. "
            "For multiple texts: List[List[List[float]]]. Order aligns with 'segments'."
        )
    )
    
    class Config:
        schema_extra = {
            "example": {
                "embeddings": [[0.1, -0.2, 0.3], [0.4, -0.1, 0.2]],
                "segments": [["First text"], ["Second text"]],
                "shape": [2, 3],
                "processing_time": 0.150
            }
        }


class SimilarityResponse(BaseModel):
    """Response model for similarity computation."""
    
    similarity_matrix: List[List[float]] = Field(
        ..., 
        description="Similarity matrix between text sets",
        example=[[0.8, 0.2], [0.3, 0.9]]
    )
    shape: Tuple[int, int] = Field(
        ..., 
        description="Shape of the similarity matrix (len(texts1), len(texts2))",
        example=[2, 2]
    )
    processing_time: float = Field(
        ..., 
        description="Time taken to process the request in seconds",
        example=0.200
    )
    
    class Config:
        schema_extra = {
            "example": {
                "similarity_matrix": [[0.85, 0.23], [0.31, 0.92]],
                "shape": [2, 2],
                "processing_time": 0.200
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""
    
    index: int = Field(
        ..., 
        description="Index of the candidate text in the original list",
        example=0
    )
    text: str = Field(
        ..., 
        description="The candidate text",
        example="Machine learning is a subset of AI"
    )
    similarity_score: float = Field(
        ..., 
        description="Similarity score between query and candidate",
        example=0.875
    )
    
    class Config:
        schema_extra = {
            "example": {
                "index": 0,
                "text": "Machine learning is a subset of AI",
                "similarity_score": 0.875
            }
        }


class SearchResponse(BaseModel):
    """Response model for similarity search."""
    
    results: List[SearchResult] = Field(
        ..., 
        description="Top similar texts ranked by similarity score",
        example=[
            {"index": 0, "text": "ML helps with predictions", "similarity_score": 0.85},
            {"index": 2, "text": "AI is artificial intelligence", "similarity_score": 0.72}
        ]
    )
    query: str = Field(
        ..., 
        description="The original query text",
        example="What is machine learning?"
    )
    processing_time: float = Field(
        ..., 
        description="Time taken to process the request in seconds",
        example=0.180
    )
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {"index": 0, "text": "ML helps with predictions", "similarity_score": 0.85},
                    {"index": 2, "text": "AI is artificial intelligence", "similarity_score": 0.72}
                ],
                "query": "What is machine learning?",
                "processing_time": 0.180
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_path: str = Field(
        ..., 
        description="Path or identifier of the loaded model",
        example="sentence-transformers/all-MiniLM-L6-v2"
    )
    device: str = Field(
        ..., 
        description="Device the model is running on",
        example="cuda:0"
    )
    embedding_dim: int = Field(
        ..., 
        description="Dimension of the generated embeddings",
        example=384
    )
    max_length: int = Field(
        ..., 
        description="Maximum token length for input texts",
        example=512
    )
    default_batch_size: int = Field(
        ..., 
        description="Default batch size for processing",
        example=32
    )
    normalize_embeddings: bool = Field(
        ..., 
        description="Whether embeddings are L2 normalized",
        example=True
    )
    gpu_available: bool = Field(
        ..., 
        description="Whether GPU acceleration is available",
        example=True
    )
    gpu_memory: Optional[int] = Field(
        None, 
        description="Total GPU memory in bytes (if available)",
        example=8589934592
    )
    
    class Config:
        schema_extra = {
            "example": {
                "model_path": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cuda:0",
                "embedding_dim": 384,
                "max_length": 512,
                "default_batch_size": 32,
                "normalize_embeddings": True,
                "gpu_available": True,
                "gpu_memory": 8589934592
            }
        }


class AvailableModelsResponse(BaseModel):
    """Response model for available models list."""
    
    available_models: dict = Field(
        ..., 
        description="Dictionary of available model identifiers and their paths",
        example={
            "all-mini": "sentence-transformers/all-MiniLM-L6-v2",
            "distilbert": "distilbert-base-uncased",
            "bert-tiny": "google/bert_uncased_L-2_H-128_A-2"
        }
    )
    current_model: Optional[str] = Field(
        None, 
        description="Currently loaded model identifier",
        example="all-mini"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "available_models": {
                    "all-mini": "sentence-transformers/all-MiniLM-L6-v2",
                    "distilbert": "distilbert-base-uncased",
                    "bert-tiny": "google/bert_uncased_L-2_H-128_A-2"
                },
                "current_model": "all-mini"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(
        ..., 
        description="Service health status",
        example="healthy"
    )
    timestamp: str = Field(
        ..., 
        description="Timestamp of the health check",
        example="2025-10-21T14:30:00Z"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-10-21T14:30:00Z"
            }
        }


class StatusResponse(BaseModel):
    """Response model for detailed service status."""
    
    status: str = Field(
        ..., 
        description="Overall service status",
        example="healthy"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether a model is currently loaded",
        example=True
    )
    model_info: Optional[ModelInfoResponse] = Field(
        None, 
        description="Information about the loaded model"
    )
    gpu_available: bool = Field(
        ..., 
        description="Whether GPU is available",
        example=True
    )
    gpu_memory_used: Optional[int] = Field(
        None, 
        description="Current GPU memory usage in bytes",
        example=1073741824
    )
    gpu_memory_total: Optional[int] = Field(
        None, 
        description="Total GPU memory in bytes",
        example=8589934592
    )
    uptime_seconds: float = Field(
        ..., 
        description="Service uptime in seconds",
        example=3600.5
    )
    timestamp: str = Field(
        ..., 
        description="Timestamp of the status check",
        example="2025-10-21T14:30:00Z"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {
                    "model_path": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "cuda:0",
                    "embedding_dim": 384
                },
                "gpu_available": True,
                "gpu_memory_used": 1073741824,
                "gpu_memory_total": 8589934592,
                "uptime_seconds": 3600.5,
                "timestamp": "2025-10-21T14:30:00Z"
            }
        }


class SuccessResponse(BaseModel):
    """Generic success response for operations that don't return data."""
    
    success: bool = Field(
        True, 
        description="Whether the operation was successful",
        example=True
    )
    message: str = Field(
        ..., 
        description="Success message",
        example="Operation completed successfully"
    )
    processing_time: Optional[float] = Field(
        None, 
        description="Time taken to process the request in seconds",
        example=0.050
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Cache cleared successfully",
                "processing_time": 0.050
            }
        }


class ErrorResponse(BaseModel):
    """Error response model for all endpoints."""
    
    error: str = Field(
        ..., 
        description="Error type or category",
        example="ValidationError"
    )
    message: str = Field(
        ..., 
        description="Detailed error message",
        example="Text input is too long"
    )
    details: Optional[Any] = Field(
        None, 
        description="Additional error details",
        example={"max_length": 100000, "received_length": 150000}
    )
    timestamp: str = Field(
        ..., 
        description="Timestamp when the error occurred",
        example="2025-10-21T14:30:00Z"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Text input is too long",
                "details": {"max_length": 100000, "received_length": 150000},
                "timestamp": "2025-10-21T14:30:00Z"
            }
        }