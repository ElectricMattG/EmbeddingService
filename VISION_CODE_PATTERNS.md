# Vision Service Code Patterns & Examples

## Document Purpose

This supplementary document provides **concrete code patterns and examples** for implementing the vision service components. It serves as a technical reference alongside the main `VISION_SERVICE_MIGRATION_PLAN.md`.

---

## 1. VisualEngine Class Structure

### 1.1 Class Definition Pattern

**Text Engine Pattern (Reference):**
```python
class EmbeddingEngine(Base):
    SMALL_MODELS = {
        'distilbert': 'distilbert-base-uncased',
        'all-mini': 'sentence-transformers/all-MiniLM-L6-v2'
    }

    def __init__(self, model_name: str = None, device: Optional[str] = None, ...):
        super().__init__()
        self.model_name = model_name or config_model or "all-mini"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._tokenizer_lock = threading.Lock()
        self._load_model()
```

**Vision Engine Pattern (Proposed):**
```python
class VisualEngine(Base):
    """GPU-accelerated visual embeddings engine with frame aggregation support."""

    VISION_MODELS = {
        # OpenCLIP variants
        'openclip-vit-b-32': ('ViT-B-32', 'openai'),
        'openclip-vit-b-16': ('ViT-B-16', 'openai'),
        'openclip-vit-l-14': ('ViT-L-14', 'openai'),

        # DINOv2 variants
        'dinov2-small': 'facebook/dinov2-small',
        'dinov2-base': 'facebook/dinov2-base',
        'dinov2-large': 'facebook/dinov2-large',
    }

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        input_size: int = None,
        batch_size: int = None,
        normalize_embeddings: bool = None,
        cache_dir: Optional[str] = None,
        temporal_pooling: str = None
    ):
        super().__init__()  # Get config and logger from Base

        # Configuration with fallbacks
        config_input_size = self._app_config.get("vision.input_size")
        config_batch_size = self._app_config.get("vision.batch_size")
        config_normalize = self._app_config.get("vision.normalize_embeddings")
        config_temporal = self._app_config.get("vision.temporal_pooling")
        config_model = self._app_config.get("vision.default_model")

        self.input_size = input_size if input_size is not None else (config_input_size or 224)
        self.default_batch_size = batch_size if batch_size is not None else (config_batch_size or 32)
        self.normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else (config_normalize if config_normalize is not None else True)
        self.temporal_pooling = temporal_pooling or config_temporal or "mean"

        # Resolve model name
        self.model_name = model_name or config_model or "openclip-vit-b-32"
        if self.model_name in self.VISION_MODELS:
            self.model_config = self.VISION_MODELS[self.model_name]
        else:
            self.model_config = self.model_name

        # Device setup (same as text)
        config_device = self._app_config.get("vision.device")
        device = device or config_device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self._logger.info(f"Initializing VisualEngine with {self.model_name} on {self.device}")

        # Set cache directory
        config_cache_dir = self._app_config.get("vision.cache_dir")
        cache_dir = cache_dir or config_cache_dir
        if cache_dir:
            os.environ['TORCH_HOME'] = cache_dir

        # Thread pool for async operations (same pattern as text)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="VisualEngine")

        # Image preprocessing lock (PIL not thread-safe)
        self._preprocess_lock = threading.Lock()

        # Load model and preprocessing
        self._load_model()

        # Model info
        self.embedding_dim = self._get_embedding_dimension()
        self._logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
```

### 1.2 Model Loading Methods

**OpenCLIP Loading:**
```python
def _load_model(self):
    """Load vision model with error handling."""
    try:
        if self.model_name.startswith('openclip'):
            import open_clip
            model_arch, pretrained = self.model_config

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_arch,
                pretrained=pretrained,
                device=self.device
            )
            self.model.eval()
            self.model_type = 'openclip'

        elif self.model_name.startswith('dinov2'):
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(self.model_config)
            self.model = AutoModel.from_pretrained(self.model_config)
            self.model.to(self.device)
            self.model.eval()
            self.model_type = 'dinov2'

    except Exception as e:
        self._logger.error(f"Failed to load model {self.model_name}: {e}")
        fallback_model = self._app_config.get("vision.fallback_model") or "openclip-vit-b-32"
        self._logger.info(f"Falling back to {fallback_model}")
        self.model_name = fallback_model
        self._load_model()  # Recursive with fallback

def _get_embedding_dimension(self) -> int:
    """Get embedding dimension by running test input."""
    test_image = Image.new('RGB', (self.input_size, self.input_size), color='red')
    with torch.no_grad():
        test_embedding = self._embed_sync([test_image])
    return test_embedding.shape[1]
```

**DINOv2 Alternative:**
```python
# Using timm for DINOv2
import timm

self.model = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',  # DINOv2 base
    pretrained=True,
    num_classes=0,  # Remove classification head
)
self.model.to(self.device)
self.model.eval()

# Create preprocessing transforms
from torchvision import transforms
self.preprocess = transforms.Compose([
    transforms.Resize(self.input_size),
    transforms.CenterCrop(self.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 2. Image Processing Pipeline

### 2.1 Base64 Decoding

```python
import base64
from io import BytesIO
from PIL import Image
from typing import List, Union

def decode_base64_image(b64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        b64_string: Base64 encoded image

    Returns:
        PIL Image in RGB format

    Raises:
        ValueError: If decoding fails or image is invalid
    """
    try:
        # Handle data URI prefix if present
        if ',' in b64_string and b64_string.startswith('data:image'):
            b64_string = b64_string.split(',')[1]

        # Decode base64
        image_data = base64.b64decode(b64_string)

        # Load as PIL Image
        image = Image.open(BytesIO(image_data))

        # Convert to RGB (handles RGBA, L, P, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

def batch_decode_images(b64_list: List[str]) -> List[Image.Image]:
    """Decode multiple base64 images with error handling."""
    images = []
    for i, b64 in enumerate(b64_list):
        try:
            img = decode_base64_image(b64)
            images.append(img)
        except ValueError as e:
            raise ValueError(f"Failed to decode image at index {i}: {e}")
    return images
```

### 2.2 Preprocessing Pipeline

```python
def _preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
    """
    Preprocess images for model input.

    Args:
        images: List of PIL Images

    Returns:
        Batched tensor ready for model
    """
    if self.model_type == 'openclip':
        # OpenCLIP preprocessing
        # Note: open_clip preprocess expects single image, not batch
        processed = []
        for img in images:
            # Preprocess function from open_clip.create_model_and_transforms
            img_tensor = self.preprocess(img)
            processed.append(img_tensor)

        # Stack into batch
        batch_tensor = torch.stack(processed)

    elif self.model_type == 'dinov2':
        # HuggingFace processor handles batching
        processed = self.processor(
            images=images,
            return_tensors='pt'
        )
        batch_tensor = processed['pixel_values']

    return batch_tensor.to(self.device)
```

### 2.3 Image Validation

```python
def validate_image(img: Image.Image, max_dimension: int = 4096) -> None:
    """
    Validate image dimensions and format.

    Args:
        img: PIL Image to validate
        max_dimension: Maximum width/height allowed

    Raises:
        ValueError: If validation fails
    """
    width, height = img.size

    if width > max_dimension or height > max_dimension:
        raise ValueError(
            f"Image dimensions ({width}x{height}) exceed maximum ({max_dimension}x{max_dimension})"
        )

    if width < 32 or height < 32:
        raise ValueError(
            f"Image dimensions ({width}x{height}) too small (minimum 32x32)"
        )

    # Image is valid RGB after decode_base64_image conversion
```

---

## 3. Embedding Generation Methods

### 3.1 Single Image Embedding

```python
def _embed_single_image(self, image: Image.Image) -> np.ndarray:
    """
    Generate embedding for a single image.

    Args:
        image: PIL Image

    Returns:
        1D numpy array embedding
    """
    # Preprocess
    img_tensor = self._preprocess_images([image])

    # Forward pass
    with torch.no_grad():
        if self.model_type == 'openclip':
            # OpenCLIP returns image features directly
            features = self.model.encode_image(img_tensor)

        elif self.model_type == 'dinov2':
            # DINOv2 via transformers - get CLS token
            outputs = self.model(img_tensor)
            features = outputs.last_hidden_state[:, 0]  # CLS token

    # Normalize if requested
    if self.normalize_embeddings:
        features = torch.nn.functional.normalize(features, p=2, dim=1)

    return features.cpu().numpy()[0]
```

### 3.2 Batch Processing

```python
def _process_batch(self, images: List[Image.Image]) -> np.ndarray:
    """
    Process a batch of images and return embeddings.

    Args:
        images: List of PIL Images

    Returns:
        2D numpy array of embeddings (n_images, embedding_dim)
    """
    # Preprocess batch
    batch_tensor = self._preprocess_images(images)

    # Forward pass
    with torch.no_grad():
        if self.model_type == 'openclip':
            embeddings = self.model.encode_image(batch_tensor)

        elif self.model_type == 'dinov2':
            outputs = self.model(batch_tensor)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token

    # Normalize if requested
    if self.normalize_embeddings:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()
```

### 3.3 Main Embed Method (Async Pattern)

```python
async def embed(
    self,
    images: Union[str, List[str]],  # Base64 encoded
    batch_size: Optional[int] = None,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generate embeddings for base64-encoded images with batching.

    Args:
        images: Single base64 string or list of base64 strings
        batch_size: Batch size for processing
        show_progress: Show progress information

    Returns:
        2D numpy array (n_images, embedding_dim)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        self._executor,
        self._embed_sync,
        images,
        batch_size,
        show_progress
    )

def _embed_sync(
    self,
    images: Union[str, List[str]],
    batch_size: Optional[int] = None,
    show_progress: bool = False
) -> np.ndarray:
    """Synchronous implementation of embed method."""
    # Handle single image
    if isinstance(images, str):
        images = [images]

    if not images:
        return np.array([]).reshape(0, self.embedding_dim)

    batch_size = batch_size or self.default_batch_size
    n_images = len(images)

    if show_progress:
        self._logger.info(f"Embedding {n_images} images")

    start_time = time.time()

    # Decode all base64 images first
    pil_images = batch_decode_images(images)

    # Validate images
    for img in pil_images:
        validate_image(img)

    # Process in batches
    embeddings_list = []
    for i in range(0, n_images, batch_size):
        batch_images = pil_images[i:i + batch_size]
        batch_embeddings = self._process_batch(batch_images)
        embeddings_list.append(batch_embeddings)

        if show_progress and i % (batch_size * 5) == 0:
            progress = (i + len(batch_images)) / n_images * 100
            self._logger.info(f"Progress: {progress:.1f}%")

    # Stack all batches
    embeddings = np.vstack(embeddings_list)

    if show_progress:
        elapsed = time.time() - start_time
        rate = n_images / elapsed
        self._logger.info(f"Completed {n_images} embeddings in {elapsed:.2f}s ({rate:.1f} images/sec)")

    return embeddings
```

---

## 4. Frame Sequence Processing

### 4.1 Temporal Pooling Methods

```python
def _temporal_mean_pooling(self, frame_embeddings: np.ndarray) -> np.ndarray:
    """Average pooling across frames."""
    return np.mean(frame_embeddings, axis=0)

def _temporal_max_pooling(self, frame_embeddings: np.ndarray) -> np.ndarray:
    """Max pooling across frames."""
    return np.max(frame_embeddings, axis=0)

def _temporal_attention_pooling(self, frame_embeddings: torch.Tensor) -> np.ndarray:
    """
    Attention-based pooling across frames.
    Uses simple self-attention mechanism.
    """
    # frame_embeddings: (n_frames, embedding_dim)

    # Compute attention scores
    # Q = K = V = frame_embeddings
    scores = torch.matmul(frame_embeddings, frame_embeddings.T)  # (n_frames, n_frames)
    attention_weights = torch.nn.functional.softmax(scores, dim=1)

    # Weighted sum
    weighted = torch.matmul(attention_weights, frame_embeddings)  # (n_frames, embedding_dim)

    # Average across frames
    aggregated = torch.mean(weighted, dim=0)  # (embedding_dim,)

    return aggregated.cpu().numpy()

def _aggregate_frames(
    self,
    frame_embeddings: np.ndarray,
    method: str = None
) -> np.ndarray:
    """
    Aggregate frame embeddings into single vector.

    Args:
        frame_embeddings: Array of shape (n_frames, embedding_dim)
        method: Pooling method ('mean', 'max', 'attention')

    Returns:
        1D numpy array (embedding_dim,)
    """
    method = method or self.temporal_pooling

    if method == 'mean':
        return self._temporal_mean_pooling(frame_embeddings)
    elif method == 'max':
        return self._temporal_max_pooling(frame_embeddings)
    elif method == 'attention':
        frame_tensor = torch.from_numpy(frame_embeddings).to(self.device)
        return self._temporal_attention_pooling(frame_tensor)
    else:
        raise ValueError(f"Unknown temporal pooling method: {method}")
```

### 4.2 Frame Sequence Embedding

```python
async def embed_frames(
    self,
    frames: List[str],  # Base64 encoded frames
    temporal_pooling: Optional[str] = None,
    batch_size: Optional[int] = None,
    return_frame_embeddings: bool = False,
    show_progress: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate embeddings for a sequence of frames with temporal aggregation.

    Args:
        frames: List of base64 encoded frames
        temporal_pooling: 'mean', 'max', or 'attention'
        batch_size: Batch size for processing frames
        return_frame_embeddings: If True, return both aggregated and per-frame
        show_progress: Show progress information

    Returns:
        If return_frame_embeddings=False: Aggregated embedding (1D array)
        If return_frame_embeddings=True: (aggregated, frame_embeddings) tuple
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        self._executor,
        self._embed_frames_sync,
        frames,
        temporal_pooling,
        batch_size,
        return_frame_embeddings,
        show_progress
    )

def _embed_frames_sync(
    self,
    frames: List[str],
    temporal_pooling: Optional[str] = None,
    batch_size: Optional[int] = None,
    return_frame_embeddings: bool = False,
    show_progress: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Synchronous implementation of embed_frames."""

    if not frames:
        return np.array([])

    # Get per-frame embeddings using standard embed
    frame_embeddings = self._embed_sync(
        frames,
        batch_size=batch_size,
        show_progress=show_progress
    )

    # Aggregate across temporal dimension
    aggregated = self._aggregate_frames(
        frame_embeddings,
        method=temporal_pooling
    )

    if return_frame_embeddings:
        return aggregated, frame_embeddings
    return aggregated
```

---

## 5. Similarity Computation

### 5.1 Image-to-Image Similarity

```python
async def similarity(
    self,
    images1: Union[str, List[str]],
    images2: Union[str, List[str]]
) -> np.ndarray:
    """
    Compute cosine similarity between two sets of images.

    Args:
        images1: First set of base64 images
        images2: Second set of base64 images

    Returns:
        Similarity matrix (len(images1), len(images2))
    """
    embeddings1 = await self.embed(images1)
    embeddings2 = await self.embed(images2)

    # Ensure normalized for cosine similarity
    if not self.normalize_embeddings:
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    return np.dot(embeddings1, embeddings2.T)
```

### 5.2 Image Search

```python
async def find_most_similar(
    self,
    query_image: str,  # Base64
    candidate_images: List[str],  # Base64
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Find most similar images to a query image.

    Args:
        query_image: Base64 query image
        candidate_images: List of base64 candidate images
        top_k: Number of results to return

    Returns:
        List of (index, similarity_score) tuples
    """
    similarities = await self.similarity([query_image], candidate_images)
    similarities = similarities[0]  # Get first row

    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((int(idx), float(similarities[idx])))

    return results
```

---

## 6. API Routes Implementation

### 6.1 Vision Embeddings Route

```python
# src/service/routes/vision_embeddings.py

from fastapi import APIRouter, HTTPException, status
from src.service.core.visual_engine_manager import visual_engine_manager
from src.service.schemas.requests import VisionEmbedRequest
from src.service.schemas.responses import VisionEmbedResponse

router = APIRouter(prefix="/vision", tags=["Vision Embeddings"])

@router.post(
    "/embeddings",
    response_model=VisionEmbedResponse,
    summary="Generate image embeddings",
    description="Generate embeddings for base64-encoded images"
)
async def generate_vision_embeddings(request: VisionEmbedRequest) -> VisionEmbedResponse:
    """
    Generate embeddings for input images with batching support.

    Args:
        request: Vision embedding request with base64 images

    Returns:
        VisionEmbedResponse with embeddings and metadata
    """
    start_time = time.time()

    try:
        # Generate embeddings using the manager
        embeddings = await visual_engine_manager.embed(
            images=request.images,
            batch_size=request.batch_size,
            show_progress=request.show_progress
        )

        # Convert to list for JSON
        embeddings_list = embeddings.tolist()
        processing_time = time.time() - start_time

        return VisionEmbedResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape),
            processing_time=processing_time,
            image_count=embeddings.shape[0]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}"
        )
```

### 6.2 Frame Sequence Route

```python
@router.post(
    "/embeddings/frames",
    response_model=VisionFrameEmbedResponse,
    summary="Generate embeddings for frame sequence",
    description="Generate embeddings for video frames with temporal aggregation"
)
async def generate_frame_embeddings(
    request: VisionFrameEmbedRequest
) -> VisionFrameEmbedResponse:
    """
    Generate embeddings for a sequence of frames.

    Args:
        request: Frame embedding request with base64 frames

    Returns:
        VisionFrameEmbedResponse with aggregated and per-frame embeddings
    """
    start_time = time.time()

    try:
        result = await visual_engine_manager.embed_frames(
            frames=request.frames,
            temporal_pooling=request.temporal_pooling,
            batch_size=request.batch_size,
            return_frame_embeddings=request.return_frame_embeddings,
            show_progress=False
        )

        if request.return_frame_embeddings:
            aggregated, frame_embeddings = result
            frame_embeddings_list = frame_embeddings.tolist()
        else:
            aggregated = result
            frame_embeddings_list = None

        aggregated_list = aggregated.tolist()
        processing_time = time.time() - start_time

        return VisionFrameEmbedResponse(
            aggregated_embedding=aggregated_list,
            frame_embeddings=frame_embeddings_list,
            frame_count=len(request.frames),
            shape=[len(request.frames), len(aggregated_list)] if frame_embeddings_list else [len(aggregated_list)],
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate frame embeddings: {str(e)}"
        )
```

---

## 7. Request/Response Schemas

### 7.1 Vision Request Models

```python
# src/service/schemas/requests.py (additions)

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
import base64

class VisionEmbedRequest(BaseModel):
    """Request model for vision embedding endpoint."""

    images: Union[str, List[str]] = Field(
        ...,
        description="Single base64 image or list of base64 images",
        example="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size for processing",
        ge=1,
        le=100,
        example=32
    )
    show_progress: bool = Field(
        False,
        description="Show progress information"
    )
    normalize: bool = Field(
        True,
        description="L2 normalize embeddings"
    )

    @validator('images')
    def validate_images(cls, v):
        """Validate base64 images."""
        if isinstance(v, str):
            # Single image
            cls._validate_single_image(v)
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Image list cannot be empty")
            if len(v) > 100:
                raise ValueError("Too many images (max 100 per request)")
            for i, img in enumerate(v):
                if not isinstance(img, str):
                    raise ValueError(f"Image at index {i} must be string")
                cls._validate_single_image(img, index=i)
        return v

    @staticmethod
    def _validate_single_image(b64_string: str, index: Optional[int] = None):
        """Validate single base64 image."""
        prefix = f"Image at index {index}: " if index is not None else ""

        if not b64_string:
            raise ValueError(f"{prefix}Image cannot be empty")

        # Check base64 format
        try:
            # Handle data URI
            if b64_string.startswith('data:image'):
                b64_string = b64_string.split(',')[1]

            # Try decode
            image_data = base64.b64decode(b64_string)

            # Check size (10MB limit)
            if len(image_data) > 10 * 1024 * 1024:
                raise ValueError(f"{prefix}Image size exceeds 10MB")

        except Exception as e:
            raise ValueError(f"{prefix}Invalid base64 encoding: {e}")

class VisionFrameEmbedRequest(BaseModel):
    """Request model for frame sequence embedding."""

    frames: List[str] = Field(
        ...,
        description="List of base64 encoded frames",
        min_items=1,
        max_items=100
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size for frame processing",
        ge=1,
        le=50
    )
    temporal_pooling: Literal['mean', 'max', 'attention'] = Field(
        'mean',
        description="Temporal aggregation method"
    )
    return_frame_embeddings: bool = Field(
        False,
        description="Return per-frame embeddings in addition to aggregated"
    )

    @validator('frames')
    def validate_frames(cls, v):
        """Validate frame sequence."""
        if len(v) < 1:
            raise ValueError("Must provide at least 1 frame")
        if len(v) > 100:
            raise ValueError("Too many frames (max 100)")

        # Validate each frame as base64 image
        for i, frame in enumerate(v):
            VisionEmbedRequest._validate_single_image(frame, index=i)

        return v
```

### 7.2 Vision Response Models

```python
# src/service/schemas/responses.py (additions)

class VisionEmbedResponse(BaseModel):
    """Response model for vision embedding generation."""

    embeddings: List[List[float]] = Field(
        ...,
        description="Generated embeddings as 2D list",
        example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    shape: Tuple[int, int] = Field(
        ...,
        description="Shape (n_images, embedding_dim)",
        example=(2, 512)
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds",
        example=0.234
    )
    image_count: int = Field(
        ...,
        description="Number of images processed",
        example=2
    )

class VisionFrameEmbedResponse(BaseModel):
    """Response model for frame sequence embedding."""

    aggregated_embedding: List[float] = Field(
        ...,
        description="Temporally aggregated embedding",
        example=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    frame_embeddings: Optional[List[List[float]]] = Field(
        None,
        description="Per-frame embeddings if requested"
    )
    frame_count: int = Field(
        ...,
        description="Number of frames processed",
        example=10
    )
    shape: Union[Tuple[int], Tuple[int, int]] = Field(
        ...,
        description="Shape of embeddings",
        example=(10, 512)
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds",
        example=0.456
    )
```

---

## 8. Visual Engine Manager

### 8.1 Manager Structure

```python
# src/service/core/visual_engine_manager.py

from src.components.visual_engine import VisualEngine
from src.service.core.exceptions import ModelNotLoadedException

class VisualEngineManager:
    """Singleton manager for VisualEngine instances."""

    _instance: Optional['VisualEngineManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'VisualEngineManager':
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'VisualEngineManager':
        """Get singleton instance."""
        return cls()

    def __init__(self):
        """Initialize manager (once)."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._engine: Optional[VisualEngine] = None
        self._current_model: Optional[str] = None
        self._model_lock = asyncio.Lock()

        # Semaphores for vision operations
        from src.config import app_conf
        self._semaphore_config = {
            'vision_operations': int(app_conf.get('service.vision_concurrency_limit') or 16),
        }
        self._semaphores = {
            'vision_operations': asyncio.Semaphore(self._semaphore_config['vision_operations']),
        }

        # Available models
        self._available_models = VisualEngine.VISION_MODELS.copy()
        self._default_model = app_conf.get('vision.default_model') or 'openclip-vit-b-32'
        self._auto_load_default = True

    async def ensure_model_loaded(self, model_name: Optional[str] = None) -> VisualEngine:
        """Ensure model is loaded."""
        async with self._model_lock:
            if model_name is None:
                if self._engine is not None:
                    return self._engine
                elif self._auto_load_default:
                    model_name = self._default_model
                else:
                    raise ModelNotLoadedException()

            if self._engine is not None and self._current_model == model_name:
                return self._engine

            await self._load_model(model_name)
            return self._engine

    async def _load_model(self, model_name: str):
        """Load vision model."""
        # Use executor for blocking model load
        loop = asyncio.get_event_loop()
        engine = await loop.run_in_executor(
            None,
            lambda: VisualEngine(model_name=model_name)
        )
        self._engine = engine
        self._current_model = model_name

    async def embed(
        self,
        images: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings with semaphore control."""
        engine = await self.ensure_model_loaded()

        async with self.get_semaphore('vision_operations'):
            return await engine.embed(
                images=images,
                batch_size=batch_size,
                show_progress=show_progress
            )

    # ... similar methods for embed_frames, similarity, etc.

# Global instance
visual_engine_manager = VisualEngineManager.get_instance()
```

---

## 9. Testing Patterns

### 9.1 Unit Test Structure

```python
# tests/test_visual_engine.py

import pytest
import numpy as np
from src.components.visual_engine import VisualEngine
from PIL import Image
import base64
from io import BytesIO

@pytest.fixture
def visual_engine():
    """Create VisualEngine instance for testing."""
    return VisualEngine(model_name='openclip-vit-b-32')

@pytest.fixture
def sample_base64_image():
    """Generate sample base64 encoded image."""
    img = Image.new('RGB', (224, 224), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    b64_string = base64.b64encode(img_bytes).decode('utf-8')
    return b64_string

@pytest.mark.asyncio
async def test_single_image_embedding(visual_engine, sample_base64_image):
    """Test embedding generation for single image."""
    embedding = await visual_engine.embed(sample_base64_image)

    assert embedding.shape == (1, visual_engine.embedding_dim)
    assert embedding.dtype == np.float32 or embedding.dtype == np.float64

    # Check normalized if configured
    if visual_engine.normalize_embeddings:
        norm = np.linalg.norm(embedding[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

@pytest.mark.asyncio
async def test_batch_image_embedding(visual_engine, sample_base64_image):
    """Test embedding generation for batch of images."""
    images = [sample_base64_image] * 10
    embeddings = await visual_engine.embed(images)

    assert embeddings.shape == (10, visual_engine.embedding_dim)

@pytest.mark.asyncio
async def test_frame_sequence_embedding(visual_engine, sample_base64_image):
    """Test frame sequence with temporal pooling."""
    frames = [sample_base64_image] * 5

    # Test mean pooling
    aggregated = await visual_engine.embed_frames(
        frames,
        temporal_pooling='mean'
    )
    assert aggregated.shape == (visual_engine.embedding_dim,)

    # Test with frame embeddings
    aggregated, frame_embs = await visual_engine.embed_frames(
        frames,
        temporal_pooling='mean',
        return_frame_embeddings=True
    )
    assert frame_embs.shape == (5, visual_engine.embedding_dim)

@pytest.mark.asyncio
async def test_similarity_computation(visual_engine, sample_base64_image):
    """Test image similarity computation."""
    img1 = [sample_base64_image]
    img2 = [sample_base64_image] * 3

    sim_matrix = await visual_engine.similarity(img1, img2)

    assert sim_matrix.shape == (1, 3)
    # Self-similarity should be ~1.0
    assert np.isclose(sim_matrix[0, 0], 1.0, atol=1e-5)
```

### 9.2 Integration Test

```python
# tests/test_vision_routes.py

from fastapi.testclient import TestClient
from src.service.main import app

client = TestClient(app)

def test_vision_embeddings_endpoint(sample_base64_image):
    """Test POST /vision/embeddings endpoint."""
    response = client.post(
        "/vision/embeddings",
        json={
            "images": [sample_base64_image],
            "batch_size": 32,
            "show_progress": False
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "embeddings" in data
    assert "shape" in data
    assert "processing_time" in data
    assert "image_count" in data

    assert data["image_count"] == 1
    assert len(data["embeddings"]) == 1

def test_frame_embeddings_endpoint(sample_base64_image):
    """Test POST /vision/embeddings/frames endpoint."""
    response = client.post(
        "/vision/embeddings/frames",
        json={
            "frames": [sample_base64_image] * 5,
            "temporal_pooling": "mean",
            "return_frame_embeddings": True
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "aggregated_embedding" in data
    assert "frame_embeddings" in data
    assert "frame_count" in data

    assert data["frame_count"] == 5
    assert len(data["frame_embeddings"]) == 5
```

---

## 10. Performance Optimization Patterns

### 10.1 GPU Memory Management

```python
def _process_batch_with_retry(self, images: List[Image.Image], max_attempts: int = 3) -> np.ndarray:
    """
    Process batch with automatic retry on GPU OOM.
    Pattern from embedding_engine.py adapted for vision.
    """
    current_batch_size = len(images)
    attempts = 0

    while True:
        try:
            # Try processing batch
            batch_embeddings = self._process_batch(images[:current_batch_size])

            # Success - process remaining if needed
            if current_batch_size < len(images):
                remaining = self._process_batch_with_retry(
                    images[current_batch_size:],
                    max_attempts
                )
                return np.vstack([batch_embeddings, remaining])

            return batch_embeddings

        except RuntimeError as e:
            if 'CUDA' in str(e) or 'out of memory' in str(e):
                attempts += 1
                if attempts >= max_attempts or current_batch_size == 1:
                    # Clear cache and raise
                    torch.cuda.empty_cache()
                    raise

                # Reduce batch size
                new_size = max(1, current_batch_size // 2)
                self._logger.warning(
                    f"GPU OOM, reducing batch from {current_batch_size} to {new_size}"
                )
                current_batch_size = new_size
            else:
                raise
```

### 10.2 Preprocessing Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _get_preprocessed_image_cached(self, img_hash: str, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Cache preprocessed images to avoid redundant decoding.
    Only use for repeated identical images.
    """
    # This would require careful implementation
    # Hash-based caching of preprocessed tensors
    pass
```

---

## Summary

This document provides concrete code patterns for implementing vision capabilities by following the established patterns in the text embedding service. Key takeaways:

1. **Preserve async/sync pattern** - Use `asyncio.get_event_loop().run_in_executor()`
2. **Reuse Base class** - Configuration and logging infrastructure works as-is
3. **Mirror manager pattern** - Singleton, lazy loading, semaphore control
4. **Follow route structure** - Request validation, error handling, response formatting
5. **Adapt not rewrite** - Image processing replaces text tokenization, but flow is identical
6. **GPU memory management** - Retry with batch reduction pattern works for vision too
7. **Testing patterns** - Unit tests for engine, integration tests for routes

**Next Step:** Begin implementation with `visual_engine.py` using OpenCLIP as the first model integration.

---

**Document Version:** 1.0
**Date:** October 24, 2025
**Status:** Technical Reference
