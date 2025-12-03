"""
GPU-accelerated embeddings engine using small BERT models with batching support.

This module provides efficient text embedding generation using pre-trained BERT models
with automatic GPU detection, batching, and memory management.
"""

import asyncio
import logging
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

from src.config_handler import Base

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class EmbeddingEngine(Base):
    """
    GPU-accelerated text embedding engine with batching support.

    Uses small, efficient BERT models for generating embeddings suitable for
    semantic similarity, clustering, and retrieval tasks.
    """

    # Available small BERT models optimized for speed/memory
    SMALL_MODELS = {
        'distilbert': 'distilbert-base-uncased',
        'minibert': 'microsoft/DialoGPT-medium',  # Fallback to distilbert if not available
        'bert-tiny': 'google/bert_uncased_L-2_H-128_A-2',
        'bert-mini': 'google/bert_uncased_L-4_H-256_A-4',
        'bert-small': 'google/bert_uncased_L-4_H-512_A-8',
        'all-mini': 'sentence-transformers/all-MiniLM-L6-v2',
        'legal-bert': 'nlpaueb/legal-bert-base-uncased'
    }

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        max_length: int = None,
        batch_size: int = None,
        normalize_embeddings: bool = None,
        cache_dir: Optional[str] = None,
        segment_overlap: int = None
    ):
        """
        Initialize the embedding engine.

        Args:
            model_name: Model identifier from SMALL_MODELS or custom model path (uses config default if None)
            device: Device to use ('cuda', 'cpu', or None for auto-detection from config)
            max_length: Maximum token length for input texts (uses config default if None)
            batch_size: Default batch size for processing (uses config default if None)
            normalize_embeddings: Whether to L2 normalize embeddings (uses config default if None)
            cache_dir: Directory to cache downloaded models (uses config default if None)
            segment_overlap: Number of tokens to overlap between segments for long texts (uses config default if None)
        """
        # Initialize base class (gives us _app_config and _logger)
        super().__init__()

        # Get configuration values with fallback defaults
        config_max_length = self._app_config.get("embedding.max_length")
        config_batch_size = self._app_config.get("embedding.batch_size")
        config_normalize = self._app_config.get("embedding.normalize_embeddings")
        config_overlap = self._app_config.get("embedding.segment_overlap")
        config_model = self._app_config.get("embedding.default_model")

        self.max_length = max_length if max_length is not None else (config_max_length or 512)
        self.default_batch_size = batch_size if batch_size is not None else (config_batch_size or 32)
        self.normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else (config_normalize if config_normalize is not None else True)
        self.segment_overlap = segment_overlap if segment_overlap is not None else (config_overlap or 50)

        # Resolve model name
        self.model_name = model_name or config_model or "all-mini"
        if self.model_name in self.SMALL_MODELS:
            self.model_path = self.SMALL_MODELS[self.model_name]
        else:
            self.model_path = self.model_name

        # Device setup
        config_device = self._app_config.get("embedding.device")
        device = device or config_device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self._logger.info(f"Initializing EmbeddingEngine with {self.model_path} on {self.device}")

        # Set cache directory
        config_cache_dir = self._app_config.get("embedding.cache_dir")
        cache_dir = cache_dir or config_cache_dir
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir

        # Initialize thread pool executor for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="EmbeddingEngine")
        # Guard non-thread-safe tokenizer operations
        self._tokenizer_lock = threading.Lock()

        # Load model and tokenizer
        self._load_model()

        # Model info
        self.embedding_dim = self._get_embedding_dimension()
        self._logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def _load_model(self):
        """Load tokenizer and model with error handling."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                do_lower_case=True,
                clean_up_tokenization_spaces=True
            )

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                output_hidden_states=False,
                return_dict=True
            )

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

        except Exception as e:
            self._logger.error(f"Failed to load model {self.model_path}: {e}")
            # Fallback to configured fallback model
            fallback_model = self._app_config.get("embedding.fallback_model") or "distilbert-base-uncased"
            self._logger.info(f"Falling back to {fallback_model}")
            self.model_path = fallback_model

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension by running a test input."""
        test_text = "Test text for dimension detection"
        with torch.no_grad():
            test_embedding = self._embed_sync([test_text])
        return test_embedding.shape[1]

    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to get sentence embeddings.

        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask to ignore padding tokens

        Returns:
            Mean-pooled embeddings
        """
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _segment_text(self, text: str, overlap: Optional[int] = None) -> List[str]:
        """
        Segment a long text into chunks that fit within max_length.

        Args:
            text: Input text to segment
            overlap: Number of tokens to overlap between segments (uses instance default if None)

        Returns:
            List of text segments
        """
        if overlap is None:
            overlap = self.segment_overlap

        # Quick tokenization to get token count (tokenizers are not thread-safe)
        with self._tokenizer_lock:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # If text fits within max_length, return as is
        if len(tokens) <= self.max_length - 2:  # Account for special tokens
            return [text]

        # Calculate effective chunk size (accounting for special tokens and overlap)
        chunk_size = self.max_length - 2  # Reserve space for [CLS] and [SEP]
        segments = []

        start_idx = 0
        while start_idx < len(tokens):
            # Calculate end index for this segment
            end_idx = min(start_idx + chunk_size, len(tokens))

            # Extract token chunk
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text (guard tokenizer access)
            with self._tokenizer_lock:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            segments.append(chunk_text)

            # Move start index for next chunk (with overlap)
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - overlap

        return segments

    def _max_pooling_across_segments(self, segment_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply max pooling across multiple segments of the same text.

        Args:
            segment_embeddings: Tensor of shape (n_segments, embedding_dim)

        Returns:
            Max-pooled embedding of shape (embedding_dim,)
        """
        # Take maximum across segments for each dimension
        max_embedding, _ = torch.max(segment_embeddings, dim=0)
        return max_embedding

    def _process_batch(self, texts: List[str]) -> np.ndarray:
        """
        Process a batch of texts and return embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings
        """
        # Tokenize (guard tokenizer access; fast tokenizers aren't thread-safe)
        with self._tokenizer_lock:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Forward pass
        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply mean pooling
        embeddings = self._mean_pooling(model_output, attention_mask)

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def _process_segmented_text(self, text: str, return_segments: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Process a potentially long text by segmenting it and using max pooling.

        Args:
            text: Input text string
            return_segments: Whether to return the text segments along with embedding

        Returns:
            If return_segments=False: Max-pooled embedding as numpy array
            If return_segments=True: Tuple of (max-pooled embedding, list of segments)
        """
        # Segment the text
        segments = self._segment_text(text)

        if len(segments) == 1:
            # Single segment - process normally
            embedding = self._process_batch(segments)[0]
            if return_segments:
                return embedding, segments
            return embedding

        # Multiple segments - process all and apply max pooling
        # Note: Process segments in batches across the whole text to make use of GPU
        segment_embeddings_list = []

        # Batch segments and process
        batch_size = self.default_batch_size
        for i in range(0, len(segments), batch_size):
            batch_segments = segments[i:i + batch_size]
            batch_embeddings = self._process_batch(batch_segments)
            segment_embeddings_list.append(batch_embeddings)

        # Concatenate all segment embeddings
        all_segment_embeddings = np.vstack(segment_embeddings_list)

        # Apply max pooling across segments using numpy for consistency
        max_pooled_embedding = np.max(all_segment_embeddings, axis=0)

        if return_segments:
            return max_pooled_embedding, segments
        return max_pooled_embedding

    def _process_flat_segments(self, flat_segments: List[str], segment_batch_size: int, show_progress: bool = False) -> np.ndarray:
        """
        Process a flat list of segments in batches and return embeddings.

        Args:
            flat_segments: List of segment strings (flattened across texts)
            segment_batch_size: Batch size to use when processing segments
            show_progress: Whether to log progress

        Returns:
            Numpy array of embeddings with shape (n_segments, embedding_dim)
        """
        if not flat_segments:
            return np.array([]).reshape(0, self.embedding_dim)

        total = len(flat_segments)
        start_time = time.time()

    # Retry/backoff loop for GPU memory errors: reduce batch size on failure
        attempts = 0
        max_attempts = 6
        current_batch_size = max(1, int(segment_batch_size))

        # Enforce a server-side maximum per-chunk to avoid sudden OOM when client requests huge segment lists
        server_max = 5000  # Hardcoded server-side limit
        current_batch_size = min(current_batch_size, server_max)

        while True:
            embeddings_parts = []
            try:
                for i in range(0, total, current_batch_size):
                    batch_segments = flat_segments[i:i + current_batch_size]
                    batch_embeddings = self._process_batch(batch_segments)
                    embeddings_parts.append(batch_embeddings)

                    if show_progress:
                        processed = min(i + current_batch_size, total)
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else float('inf')
                        self._logger.info(f"Segments processed: {processed}/{total} ({rate:.1f} segments/sec)")

                # Success
                return np.vstack(embeddings_parts)

            except Exception as e:
                # Check if this looks like a CUDA/GPU error
                err_txt = str(e).lower()
                if any(key in err_txt for key in ["cuda", "nvml", "cudacachingallocator", "out of memory"]):
                    attempts += 1
                    if current_batch_size == 1 or attempts >= max_attempts:
                        # Try to clear cache and final attempt
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        # Raise a GPUMemoryException via handler
                        raise RuntimeError(f"CUDA error during embedding generation: {e}")
                    # Reduce batch size and retry
                    new_batch_size = max(1, current_batch_size // 2)
                    self._logger.warning(
                        f"GPU memory error while processing segments: {e}. Reducing segment_batch_size from {current_batch_size} to {new_batch_size} and retrying (attempt {attempts}/{max_attempts})."
                    )
                    current_batch_size = new_batch_size
                    # small backoff
                    continue
                else:
                    # Some other error -> re-raise
                    raise

    async def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        use_segmentation: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for input texts with automatic batching and segmentation.

        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for processing (uses default if None)
            show_progress: Whether to show progress information
            use_segmentation: Whether to use text segmentation for long texts

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            texts,
            batch_size,
            show_progress,
            use_segmentation
        )

    def _embed_sync(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        use_segmentation: bool = True
    ) -> np.ndarray:
        """
        Synchronous implementation of embed method.
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        batch_size = batch_size or self.default_batch_size
        n_texts = len(texts)

        if show_progress:
            self._logger.info(f"Embedding {n_texts} texts (segmentation={'ON' if use_segmentation else 'OFF'})")

        start_time = time.time()
        embeddings_list = []

        if use_segmentation:
            # Multi-text segmentation batching:
            # 1) Segment all texts and collect flat list of segments
            # 2) Batch-process segments across texts on the GPU
            # 3) Re-aggregate per-text with max pooling across each text's segments
            flat_segments = []
            segments_per_text = []

            for text in texts:
                segments = self._segment_text(text)
                segments_per_text.append(len(segments))
                flat_segments.extend(segments)

            # Process all segments in batches using requested batch_size
            segment_batch_size = batch_size or self.default_batch_size
            all_segment_embeddings = self._process_flat_segments(flat_segments, segment_batch_size, show_progress)

            # Re-aggregate per text
            idx = 0
            for j, seg_count in enumerate(segments_per_text):
                if seg_count == 0:
                    embeddings_list.append(np.zeros((self.embedding_dim,), dtype=float))
                elif seg_count == 1:
                    embeddings_list.append(all_segment_embeddings[idx])
                    idx += 1
                else:
                    seg_embs = all_segment_embeddings[idx:idx + seg_count]
                    # Max pooling across the segments
                    embeddings_list.append(np.max(seg_embs, axis=0))
                    idx += seg_count
                if show_progress and (j + 1) % 10 == 0:
                    progress = (j + 1) / n_texts * 100
                    elapsed = time.time() - start_time
                    eta = elapsed / (j + 1) * n_texts - elapsed
                    self._logger.info(f"Progress: {progress:.1f}% - ETA: {eta:.1f}s")

            # Stack all embeddings
            embeddings = np.stack(embeddings_list)
        else:
            # Original batched processing (truncation only)
            for i in range(0, n_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._process_batch(batch_texts)
                embeddings_list.append(batch_embeddings)

                if show_progress and i % (batch_size * 10) == 0:
                    progress = (i + len(batch_texts)) / n_texts * 100
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + len(batch_texts)) * n_texts - elapsed
                    self._logger.info(f"Progress: {progress:.1f}% - ETA: {eta:.1f}s")

            # Concatenate all embeddings
            embeddings = np.vstack(embeddings_list)

        if show_progress:
            elapsed = time.time() - start_time
            rate = n_texts / elapsed
            self._logger.info(f"Completed {n_texts} embeddings in {elapsed:.2f}s ({rate:.1f} texts/sec)")

        return embeddings

    async def embed_with_segments(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[List[str]]]]:
        """
        Generate embeddings and return the text segments used.

        Args:
            texts: Single text string or list of text strings
            show_progress: Whether to show progress information

        Returns:
            If single text: Tuple of (embedding, segments_list)
            If multiple texts: Tuple of (embeddings_array, list_of_segments_lists)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_with_segments_sync,
            texts,
            show_progress
        )

    def _embed_with_segments_sync(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> Union[Tuple[np.ndarray, List[str], List[np.ndarray]], Tuple[np.ndarray, List[List[str]], List[List[np.ndarray]]]]:
        """
        Synchronous implementation of embed_with_segments method.
        """
        # Handle single string input
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim), []

        n_texts = len(texts)

        if show_progress:
            self._logger.info(f"Embedding {n_texts} texts with segmentation info")

        start_time = time.time()
        embeddings_list = []
        all_segments_lists = []

        # Process segmentation across texts in batches:
        flat_segments = []
        segments_per_text = []
        for text in texts:
            segments = self._segment_text(text)
            segments_per_text.append(segments)
            flat_segments.extend(segments)

        # Batch process all segments
        segment_batch_size = self.default_batch_size
        all_segment_embeddings = self._process_flat_segments(flat_segments, segment_batch_size, show_progress)

        # Reconstruct per-text embeddings and segment lists
        idx = 0
        per_text_segment_embeddings = []
        for segments in segments_per_text:
            seg_count = len(segments)
            if seg_count == 0:
                embeddings_list.append(np.zeros((self.embedding_dim,), dtype=float))
                all_segments_lists.append([])
                per_text_segment_embeddings.append(np.zeros((0, self.embedding_dim), dtype=float))
            elif seg_count == 1:
                embeddings_list.append(all_segment_embeddings[idx])
                all_segments_lists.append(segments)
                per_text_segment_embeddings.append(all_segment_embeddings[idx:idx+1])
                idx += 1
            else:
                seg_embs = all_segment_embeddings[idx:idx + seg_count]
                embeddings_list.append(np.max(seg_embs, axis=0))
                all_segments_lists.append(segments)
                per_text_segment_embeddings.append(seg_embs)
                idx += seg_count

            # progress already handled during re-aggregation using 'j'

        # Stack all embeddings
        embeddings = np.stack(embeddings_list)

        if show_progress:
            elapsed = time.time() - start_time
            rate = n_texts / elapsed
            total_segments = sum(len(segments) for segments in all_segments_lists)
            self._logger.info(f"Completed {n_texts} embeddings ({total_segments} total segments) in {elapsed:.2f}s ({rate:.1f} texts/sec)")

        if single_text:
            return embeddings[0], all_segments_lists[0], per_text_segment_embeddings[0]
        return embeddings, all_segments_lists, per_text_segment_embeddings

    async def similarity(self, texts1: Union[str, List[str]], texts2: Union[str, List[str]]) -> np.ndarray:
        """
        Compute cosine similarity between two sets of texts.

        Args:
            texts1: First set of texts
            texts2: Second set of texts

        Returns:
            Similarity matrix with shape (len(texts1), len(texts2))
        """
        embeddings1 = await self.embed(texts1)
        embeddings2 = await self.embed(texts2)

        # Ensure embeddings are normalized for cosine similarity
        if not self.normalize_embeddings:
            embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        return np.dot(embeddings1, embeddings2.T)

    async def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Find the most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top matches to return

        Returns:
            List of (index, text, similarity_score) tuples
        """
        similarities = await self.similarity([query], candidates)
        similarities = similarities[0]  # Get the first row

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((idx, candidates[idx], similarities[idx]))

        return results

    async def quick_embed(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None,
        use_segmentation: bool = True
    ) -> np.ndarray:
        """
        Quick embedding generation with optional model switching.

        Args:
            texts: Text(s) to embed
            model_name: Optional model name to switch to
            use_segmentation: Whether to use text segmentation

        Returns:
            Embeddings array
        """
        # Note: model switching would require reloading, which is expensive
        # For now, just use the current model and log a warning if different model requested
        if model_name and model_name not in [self.model_path, self.model_name]:
            self._logger.warning(
                f"Model switching from {self.model_path} to {model_name} not implemented in quick_embed. "
                f"Using current model."
            )

        return await self.embed(texts, use_segmentation=use_segmentation)

    async def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_model_info_sync
        )

    def _get_model_info_sync(self) -> dict:
        """Synchronous implementation of get_model_info."""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'default_batch_size': self.default_batch_size,
            'normalize_embeddings': self.normalize_embeddings,
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        }

    async def clear_cache(self):
        """Clear GPU cache to free memory."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._clear_cache_sync
        )

    def _clear_cache_sync(self):
        """Synchronous implementation of clear_cache."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (AttributeError, NameError):
            # torch might be None during shutdown
            pass

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.clear_cache()
        except:
            pass

        # Shutdown the thread pool executor
        if hasattr(self, '_executor'):
            try:
                self._executor.shutdown(wait=False)
            except:
                pass

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test texts
    test_texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The dog ran in the park.",
        "Artificial intelligence is transforming technology.",
        "Machine learning algorithms process data efficiently."
    ]

    print("Initializing EmbeddingEngine...")
    engine = EmbeddingEngine(model_name='all-mini', show_progress=True)

    print(f"\nModel info: {engine.get_model_info()}")

    print(f"\nGenerating embeddings for {len(test_texts)} texts...")
    embeddings = engine.embed(test_texts, show_progress=True)
    print(f"Generated embeddings shape: {embeddings.shape}")

    print("\nFinding most similar texts to query...")
    query = "A cat lying on a carpet"
    similar = engine.find_most_similar(query, test_texts, top_k=3)

    print(f"\nQuery: '{query}'")
    print("Most similar texts:")
    for idx, text, score in similar:
        print(f"  {score:.4f}: {text}")

    print("\nComputing similarity matrix...")
    sim_matrix = engine.similarity(test_texts[:2], test_texts[2:])
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Sample similarities:\n{sim_matrix}")
