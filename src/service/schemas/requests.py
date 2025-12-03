"""
Pydantic request models for the FastAPI Embedding Service.

These models define the structure and validation for incoming requests
to all service endpoints.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator


class EmbedRequest(BaseModel):
    """Request model for the main embedding endpoint."""

    texts: Union[str, List[str]] = Field(
        ...,
        description="Single text string or list of text strings to embed",
        example=["Hello world", "This is a test sentence"]
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size for processing (uses default if None)",
        ge=1,
        le=1000,
        example=32
    )
    show_progress: bool = Field(
        False,
        description="Whether to show progress information",
        example=False
    )
    use_segmentation: bool = Field(
        True,
        description="Whether to use text segmentation for long texts",
        example=True
    )
    include_segment_embeddings: bool = Field(
        False,
        description="If true and segmentation is enabled, include per-segment embeddings in the response.",
        example=False
    )

    @validator('texts')
    def validate_texts(cls, v):
        """Validate text inputs."""
        if isinstance(v, str):
            if len(v.strip()) == 0:
                raise ValueError("Text cannot be empty")
            if len(v) > 100000:  # 100k character limit
                raise ValueError("Text too long (max 100,000 characters)")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Text list cannot be empty")
            if len(v) > 1000:  # Max 1000 texts per request
                raise ValueError("Too many texts (max 1,000 per request)")
            for text in v:
                if not isinstance(text, str):
                    raise ValueError("All items in text list must be strings")
                if len(text.strip()) == 0:
                    raise ValueError("Text cannot be empty")
                if len(text) > 100000:
                    raise ValueError("Text too long (max 100,000 characters)")
        return v


class QuickEmbedRequest(BaseModel):
    """Request model for the quick embedding endpoint."""

    texts: Union[str, List[str]] = Field(
        ...,
        description="Single text string or list of text strings to embed",
        example="Quick embedding test"
    )
    model_name: str = Field(
        'all-mini',
        description="Model to use for embedding generation",
        example="all-mini"
    )
    use_segmentation: bool = Field(
        True,
        description="Whether to use text segmentation for long texts",
        example=True
    )

    @validator('texts')
    def validate_texts(cls, v):
        """Validate text inputs (same as EmbedRequest)."""
        if isinstance(v, str):
            if len(v.strip()) == 0:
                raise ValueError("Text cannot be empty")
            if len(v) > 50000:  # Smaller limit for quick requests
                raise ValueError("Text too long for quick embedding (max 50,000 characters)")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Text list cannot be empty")
            if len(v) > 100:  # Smaller limit for quick requests
                raise ValueError("Too many texts for quick embedding (max 100 per request)")
            for text in v:
                if not isinstance(text, str):
                    raise ValueError("All items in text list must be strings")
                if len(text.strip()) == 0:
                    raise ValueError("Text cannot be empty")
                if len(text) > 50000:
                    raise ValueError("Text too long for quick embedding (max 50,000 characters)")
        return v

    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name against available models."""
        # Note: We'll validate against SMALL_MODELS in the route handler
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()


class SimilarityRequest(BaseModel):
    """Request model for similarity computation."""

    texts1: Union[str, List[str]] = Field(
        ...,
        description="First set of texts for similarity comparison",
        example=["Hello world", "Good morning"]
    )
    texts2: Union[str, List[str]] = Field(
        ...,
        description="Second set of texts for similarity comparison",
        example=["Hi there", "Good evening"]
    )

    @validator('texts1', 'texts2')
    def validate_texts(cls, v):
        """Validate text inputs."""
        if isinstance(v, str):
            if len(v.strip()) == 0:
                raise ValueError("Text cannot be empty")
            if len(v) > 50000:
                raise ValueError("Text too long for similarity (max 50,000 characters)")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Text list cannot be empty")
            if len(v) > 500:  # Reasonable limit for similarity matrices
                raise ValueError("Too many texts for similarity (max 500 per set)")
            for text in v:
                if not isinstance(text, str):
                    raise ValueError("All items in text list must be strings")
                if len(text.strip()) == 0:
                    raise ValueError("Text cannot be empty")
                if len(text) > 50000:
                    raise ValueError("Text too long for similarity (max 50,000 characters)")
        return v


class SearchRequest(BaseModel):
    """Request model for similarity search."""

    query: str = Field(
        ...,
        description="Query text to find similar texts for",
        example="What is machine learning?"
    )
    candidates: List[str] = Field(
        ...,
        description="List of candidate texts to search through",
        example=["AI is artificial intelligence", "ML helps with predictions", "Deep learning uses neural networks"]
    )
    top_k: int = Field(
        5,
        description="Number of top matches to return",
        ge=1,
        le=100,
        example=5
    )

    @validator('query')
    def validate_query(cls, v):
        """Validate query text."""
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        if len(v) > 10000:
            raise ValueError("Query too long (max 10,000 characters)")
        return v.strip()

    @validator('candidates')
    def validate_candidates(cls, v):
        """Validate candidate texts."""
        if len(v) == 0:
            raise ValueError("Candidates list cannot be empty")
        if len(v) > 10000:
            raise ValueError("Too many candidates (max 10,000)")
        for candidate in v:
            if not isinstance(candidate, str):
                raise ValueError("All candidates must be strings")
            if len(candidate.strip()) == 0:
                raise ValueError("Candidate text cannot be empty")
            if len(candidate) > 50000:
                raise ValueError("Candidate text too long (max 50,000 characters)")
        return v


class ModelLoadRequest(BaseModel):
    """Request model for loading a specific model."""

    model_name: str = Field(
        ...,
        description="Model identifier to load",
        example="all-mini"
    )
    device: Optional[str] = Field(
        None,
        description="Device to use ('cuda', 'cpu', or None for auto-detection)",
        example="cuda"
    )
    max_length: Optional[int] = Field(
        None,
        description="Maximum token length for input texts",
        ge=32,
        le=2048,
        example=512
    )
    batch_size: Optional[int] = Field(
        None,
        description="Default batch size for processing",
        ge=1,
        le=256,
        example=32
    )
    normalize_embeddings: Optional[bool] = Field(
        None,
        description="Whether to L2 normalize embeddings",
        example=True
    )

    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @validator('device')
    def validate_device(cls, v):
        """Validate device specification."""
        if v is not None:
            v = v.strip().lower()
            if v not in ['cuda', 'cpu']:
                raise ValueError("Device must be 'cuda' or 'cpu'")
        return v
