

import asyncio
from typing import List, Tuple, Union

import numpy as np

from src.components.embedding_engine import EmbeddingEngine


# Convenience function for quick embedding generation
async def quick_embed(
    texts: Union[str, List[str]],
    model_name: str = 'all-mini',
    use_segmentation: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Quick embedding generation without persistent model loading.

    Args:
        texts: Text(s) to embed
        model_name: Model to use
        use_segmentation: Whether to use text segmentation for long texts
        **kwargs: Additional arguments for EmbeddingEngine

    Returns:
        Numpy array of embeddings
    """
    engine = EmbeddingEngine(model_name=model_name, **kwargs)
    return await engine.embed(texts, use_segmentation=use_segmentation)


async def quick_embed_with_segments(
    texts: Union[str, List[str]],
    model_name: str = 'all-mini',
    **kwargs
) -> Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[List[str]]]]:
    """
    Quick embedding generation with segment information.

    Args:
        texts: Text(s) to embed
        model_name: Model to use
        **kwargs: Additional arguments for EmbeddingEngine

    Returns:
        If single text: Tuple of (embedding, segments_list)
        If multiple texts: Tuple of (embeddings_array, list_of_segments_lists)
    """
    engine = EmbeddingEngine(model_name=model_name, **kwargs)
    return await engine.embed_with_segments(texts)
