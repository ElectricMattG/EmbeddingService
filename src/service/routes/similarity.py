"""
Similarity computation routes for the FastAPI Embedding Service.

Provides endpoints for computing similarity between text sets and performing
similarity-based search operations.
"""

import time
from typing import List, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from src.service.core.engine_manager import embedding_manager
from src.service.core.exceptions import (ConcurrencyLimitException,
                                         EmbeddingProcessingException,
                                         GPUMemoryException,
                                         ModelNotLoadedException,
                                         ValidationException)
from src.service.schemas.requests import SearchRequest, SimilarityRequest
from src.service.schemas.responses import (SearchResponse, SearchResult,
                                           SimilarityResponse)
from src.service.utils.serialization import (numpy_matrix_to_list,
                                             validate_embedding_shape)
from src.service.utils.validation import validate_similarity_inputs

# Create router
router = APIRouter(tags=["Similarity Operations"])


@router.post(
    "/compute",
    response_model=SimilarityResponse,
    summary="Compute similarity matrix",
    description="Compute cosine similarity matrix between two sets of texts using the currently loaded model.",
    responses={
        200: {
            "description": "Similarity matrix computed successfully",
            "model": SimilarityResponse
        },
        422: {
            "description": "Invalid input validation",
            "content": {
                "application/json": {
                    "example": {
                        "error": "ValidationError",
                        "message": "Too many texts for similarity computation",
                        "details": {"max_texts": 500, "received": 750}
                    }
                }
            }
        },
        429: {
            "description": "Too many concurrent requests"
        },
        503: {
            "description": "No model is currently loaded"
        },
        507: {
            "description": "Insufficient GPU memory"
        },
        500: {
            "description": "Similarity computation failed"
        }
    }
)
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Compute cosine similarity matrix between two sets of texts.

    This endpoint generates embeddings for both text sets and computes the
    cosine similarity matrix between them. The result is a matrix where
    element [i,j] represents the similarity between texts1[i] and texts2[j].

    Args:
        request: Similarity request with two sets of texts

    Returns:
        SimilarityResponse: Similarity matrix with shape and processing time

    Raises:
        ModelNotLoadedException: If no model is currently loaded
        ValidationException: If input validation fails
        EmbeddingProcessingException: If similarity computation fails
        GPUMemoryException: If GPU memory is insufficient
        ConcurrencyLimitException: If service is at capacity
    """
    start_time = time.time()

    try:
        # Validate inputs
        validate_similarity_inputs(request.texts1, request.texts2)

        # Compute similarity matrix using the manager
        similarity_matrix = await embedding_manager.similarity(
            texts1=request.texts1,
            texts2=request.texts2
        )

        # Convert numpy array to list for JSON serialization
        validate_embedding_shape(similarity_matrix)
        similarity_list = numpy_matrix_to_list(similarity_matrix)
        processing_time = time.time() - start_time

        return SimilarityResponse(
            similarity_matrix=similarity_list,
            shape=list(similarity_matrix.shape),
            processing_time=processing_time
        )

    except (ModelNotLoadedException, ValidationException, EmbeddingProcessingException,
            GPUMemoryException, ConcurrencyLimitException):
        raise
    except ValueError as e:
        raise ValidationException(detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error computing similarity: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Find most similar texts",
    description="Find the most similar texts to a query from a list of candidates using semantic similarity.",
    responses={
        200: {
            "description": "Similar texts found successfully",
            "model": SearchResponse
        },
        422: {
            "description": "Invalid input validation",
            "content": {
                "application/json": {
                    "example": {
                        "error": "ValidationError",
                        "message": "Too many candidate texts",
                        "details": {"max_candidates": 10000, "received": 15000}
                    }
                }
            }
        },
        429: {
            "description": "Too many concurrent requests"
        },
        503: {
            "description": "No model is currently loaded"
        },
        507: {
            "description": "Insufficient GPU memory"
        },
        500: {
            "description": "Similarity search failed"
        }
    }
)
async def find_similar_texts(request: SearchRequest) -> SearchResponse:
    """
    Find the most similar texts to a query from a list of candidates.

    This endpoint performs semantic similarity search by:
    1. Generating embeddings for the query and all candidates
    2. Computing cosine similarities between query and candidates
    3. Returning the top-k most similar candidates with scores

    Args:
        request: Search request with query, candidates, and top_k

    Returns:
        SearchResponse: Top similar texts with scores, query, and processing time

    Raises:
        ModelNotLoadedException: If no model is currently loaded
        ValidationException: If input validation fails
        EmbeddingProcessingException: If similarity search fails
        GPUMemoryException: If GPU memory is insufficient
        ConcurrencyLimitException: If service is at capacity
    """
    start_time = time.time()

    try:
        # Validate top_k against number of candidates
        if request.top_k > len(request.candidates):
            raise ValidationException(
                detail=f"top_k ({request.top_k}) cannot be larger than number of candidates ({len(request.candidates)})"
            )

        # Perform similarity search using the manager
        similar_results = await embedding_manager.find_most_similar(
            query=request.query,
            candidates=request.candidates,
            top_k=request.top_k
        )

        # Convert results to SearchResult objects
        search_results = []
        for index, text, score in similar_results:
            search_results.append(SearchResult(
                index=index,
                text=text,
                similarity_score=float(score)  # Ensure JSON serialization
            ))

        processing_time = time.time() - start_time

        return SearchResponse(
            results=search_results,
            query=request.query,
            processing_time=processing_time
        )

    except (ModelNotLoadedException, ValidationException, EmbeddingProcessingException,
            GPUMemoryException, ConcurrencyLimitException):
        raise
    except ValueError as e:
        raise ValidationException(detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in similarity search: {str(e)}"
        )


@router.post(
    "/batch-search",
    summary="Batch similarity search",
    description="Perform similarity search for multiple queries simultaneously for improved efficiency.",
    responses={
        200: {
            "description": "Batch search completed successfully"
        },
        422: {
            "description": "Invalid input validation"
        },
        429: {
            "description": "Too many concurrent requests"
        },
        503: {
            "description": "No model is currently loaded"
        },
        507: {
            "description": "Insufficient GPU memory"
        },
        500: {
            "description": "Batch search failed"
        }
    }
)
async def batch_similarity_search(
    queries: List[str],
    candidates: List[str],
    top_k: int = 5
):
    """
    Perform similarity search for multiple queries simultaneously.

    This endpoint is more efficient than multiple individual search requests
    when you need to search multiple queries against the same candidate set.

    Args:
        queries: List of query texts to search for
        candidates: List of candidate texts to search through
        top_k: Number of top matches to return for each query

    Returns:
        Dict with search results for each query

    Raises:
        ModelNotLoadedException: If no model is currently loaded
        ValidationException: If input validation fails
        EmbeddingProcessingException: If batch search fails
        GPUMemoryException: If GPU memory is insufficient
        ConcurrencyLimitException: If service is at capacity
    """
    start_time = time.time()

    try:
        # Input validation
        if not queries:
            raise ValidationException(detail="Queries list cannot be empty")
        if not candidates:
            raise ValidationException(detail="Candidates list cannot be empty")
        if len(queries) > 100:
            raise ValidationException(
                detail=f"Too many queries for batch search (max 100, received {len(queries)})"
            )
        if len(candidates) > 10000:
            raise ValidationException(
                detail=f"Too many candidates for batch search (max 10,000, received {len(candidates)})"
            )
        if top_k > len(candidates):
            raise ValidationException(
                detail=f"top_k ({top_k}) cannot be larger than number of candidates ({len(candidates)})"
            )

        # Validate individual query and candidate lengths
        for i, query in enumerate(queries):
            if len(query.strip()) == 0:
                raise ValidationException(detail=f"Query {i} cannot be empty")
            if len(query) > 10000:
                raise ValidationException(detail=f"Query {i} too long (max 10,000 characters)")

        for i, candidate in enumerate(candidates):
            if len(candidate.strip()) == 0:
                raise ValidationException(detail=f"Candidate {i} cannot be empty")
            if len(candidate) > 50000:
                raise ValidationException(detail=f"Candidate {i} too long (max 50,000 characters)")

        # Perform batch similarity search
        batch_results = {}

        for i, query in enumerate(queries):
            # Use the existing find_most_similar method for each query
            similar_results = await embedding_manager.find_most_similar(
                query=query,
                candidates=candidates,
                top_k=top_k
            )

            # Convert results to serializable format
            search_results = []
            for index, text, score in similar_results:
                search_results.append({
                    'index': index,
                    'text': text,
                    'similarity_score': float(score)
                })

            batch_results[f'query_{i}'] = {
                'query': query,
                'results': search_results
            }

        processing_time = time.time() - start_time

        return {
            'batch_results': batch_results,
            'total_queries': len(queries),
            'candidates_count': len(candidates),
            'top_k': top_k,
            'processing_time': processing_time,
            'timestamp': time.time()
        }

    except (ModelNotLoadedException, ValidationException, EmbeddingProcessingException,
            GPUMemoryException, ConcurrencyLimitException):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in batch similarity search: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get similarity operation statistics",
    description="Get statistics about similarity operations and performance metrics.",
    responses={
        200: {
            "description": "Successfully retrieved similarity statistics"
        },
        429: {
            "description": "Too many concurrent requests"
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def get_similarity_stats():
    """
    Get statistics and performance metrics for similarity operations.

    Returns information about:
    - Current semaphore usage for similarity requests
    - Model performance characteristics
    - Resource utilization for similarity computations

    Returns:
        Dict containing similarity operation statistics

    Raises:
        ConcurrencyLimitException: If service is at capacity
    """
    async with embedding_manager.get_semaphore('health_checks'):
        try:
            semaphore_status = embedding_manager.semaphore_status

            stats = {
                'similarity_semaphore': semaphore_status.get('similarity_requests', {}),
                'embedding_semaphore': semaphore_status.get('embedding_requests', {}),
                'gpu_semaphore': semaphore_status.get('gpu_operations', {}),
                'model_loaded': embedding_manager.is_model_loaded,
                'current_model': embedding_manager.current_model,
                'uptime_seconds': embedding_manager.uptime_seconds,
                'timestamp': time.time()
            }

            # Add model-specific stats if available
            if embedding_manager.is_model_loaded:
                try:
                    model_info = await embedding_manager.get_model_info()
                    stats['model_stats'] = {
                        'embedding_dim': model_info['embedding_dim'],
                        'device': model_info['device'],
                        'normalize_embeddings': model_info['normalize_embeddings']
                    }
                except ModelNotLoadedException:
                    stats['model_stats'] = None
            else:
                stats['model_stats'] = None

            # Add GPU memory info
            gpu_info = await embedding_manager.get_gpu_memory_info()
            stats['gpu_stats'] = gpu_info

            return stats

        except ConcurrencyLimitException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve similarity statistics: {str(e)}"
            )
