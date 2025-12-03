# Vision Service Migration Plan

## Executive Summary

This document outlines a comprehensive plan to adapt the existing `EmbeddingService` repository for vision/image embedding capabilities. The analysis covers all components that would need modification, new dependencies, architectural patterns to preserve, and implementation strategies for the visual encoder.

**Target Features:**
- Visual embeddings from RGB images (single or batches)
- Base64-encoded image input support
- Frame-level and aggregated embeddings for video/sequences
- Similarity computations between images
- Support for OpenCLIP (vision) and DINOv2 models
- Same async/sync pattern as text embeddings
- GPU acceleration with memory management

---

## 1. Core Engine Component

### 1.1 New File: `src/components/visual_engine.py`

**Pattern to Follow:** `src/components/embedding_engine.py` (654 lines)

**Key Adaptations Required:**

#### Model Support
- **Text Engine:** Uses HuggingFace transformers (BERT variants, sentence-transformers)
- **Visual Engine:** Needs:
  - OpenCLIP vision models (multiple sizes: ViT-B-32, ViT-B-16, ViT-L-14)
  - DINOv2 with registers (dinov2-small, dinov2-base, dinov2-large)
  - Model zoo dictionary similar to `SMALL_MODELS`

#### Input Processing
- **Text Engine:**
  - Tokenization with AutoTokenizer
  - Text segmentation for long documents (token-based chunking with overlap)
  - Padding and attention masks
- **Visual Engine:**
  - Base64 decode → PIL Image → RGB conversion
  - Image preprocessing (resize, normalize, center crop)
  - Handle variable resolution inputs
  - Frame batching for multi-image sequences
  - Image "segmentation" could mean patch extraction or multi-scale processing

#### Embedding Generation
- **Text Engine:**
  - Mean pooling over token embeddings
  - Max pooling across text segments
  - L2 normalization optional
- **Visual Engine:**
  - CLS token extraction (for ViT-based models)
  - Global average pooling (for convolutional features)
  - Temporal pooling for frame sequences (mean, max, or attention-based)
  - L2 normalization for similarity tasks

#### Batching Strategy
- **Text Engine:** Batch by number of texts with max_length truncation
- **Visual Engine:** Batch by number of images with fixed or adaptive resolution

#### Memory Management
- **Text Engine:** Retry logic with batch size reduction on GPU OOM
- **Visual Engine:** Same pattern + consider image downsampling on OOM

---

## 2. Configuration Updates

### 2.1 File: `src/config.py`

**Current Structure:**
```python
defaults = {
    "embedding": {
        "default_model": "all-mini",
        "batch_size": 1024,
        "max_length": 512,
        "normalize_embeddings": True,
        "device": 'cuda',
        ...
    },
    "service": {
        "max_text_length": 8192,
        "max_batch_size": 1024,
        ...
    }
}
```

**Required Changes:**

```python
defaults = {
    "embedding": { ... },  # Keep existing for backward compatibility

    "vision": {
        # Model configuration
        "default_model": "openclip-vit-b-32",  # or "dinov2-small"
        "cache_dir": None,

        # Performance settings
        "batch_size": 32,  # Lower than text due to image size
        "input_size": 224,  # Default image resolution
        "normalize_embeddings": True,

        # Device settings
        "device": 'cuda',
        "gpu_memory_fraction": 0.8,

        # Processing settings
        "show_progress": False,
        "use_multi_scale": False,  # Multi-resolution processing
        "temporal_pooling": "mean",  # For frame sequences: mean/max/attention
        "fallback_model": "openclip-vit-b-32",
    },

    "service": {
        # ... existing text limits ...

        # Vision-specific limits
        "max_image_size_mb": 10,  # Per image
        "max_image_dimension": 4096,  # Width/height
        "max_frame_batch": 100,  # Max frames per request
        "vision_concurrency_limit": 16,  # Separate from text
    }
}
```

### 2.2 Base Class Pattern

**No changes needed** - `src/config_handler.py` Base class pattern works for both:
- `EmbeddingEngine(Base)` → inherits config and logger
- `VisualEngine(Base)` → same pattern

---

## 3. Service Layer Updates

### 3.1 New Manager: `src/service/core/visual_engine_manager.py`

**Pattern:** Mirror `engine_manager.py` (454 lines)

**Key Responsibilities:**
- Singleton pattern for VisualEngine instance management
- Lazy loading with model switching
- Semaphore-based concurrency control (separate from text)
- GPU memory management
- Available vision models registry

**Differences from Text Manager:**
- Track loaded vision model separately
- Different semaphore limits (vision models are larger/slower)
- Image preprocessing validation before engine call

### 3.2 Route Updates

#### New Routes File: `src/service/routes/vision_embeddings.py`

**Endpoints to Implement:**

```
POST /vision/embeddings
- Input: Single base64 image or list of base64 images
- Output: Embeddings with shape and processing time
- Support: batch_size, show_progress, temporal_pooling

POST /vision/embeddings/frames
- Input: List of base64 frames (video sequence)
- Output: Frame-level embeddings + aggregated embedding
- Support: temporal pooling method selection

POST /vision/similarity
- Input: Two sets of images (or image vs text for CLIP)
- Output: Similarity matrix

POST /vision/search
- Input: Query image + candidate images
- Output: Top-K most similar images with scores
```

#### Similarity Routes: `src/service/routes/similarity.py`

**Option 1:** Keep text-only in existing file, create `vision_similarity.py`
**Option 2:** Extend existing routes to support vision (add `input_type` parameter)

**Recommendation:** Separate files for clarity - vision and text have different preprocessing pipelines.

---

## 4. Schema Updates

### 4.1 Request Schemas: `src/service/schemas/requests.py`

**New Classes Needed:**

```python
class VisionEmbedRequest(BaseModel):
    images: Union[str, List[str]]  # Base64 encoded
    batch_size: Optional[int] = None
    show_progress: bool = False
    temporal_pooling: Optional[str] = "mean"  # For frame sequences
    normalize: bool = True

    @validator('images')
    def validate_images(cls, v):
        # Validate base64 format
        # Check image size limits
        # Verify decodable
        ...

class VisionFrameEmbedRequest(BaseModel):
    frames: List[str]  # List of base64 encoded frames
    batch_size: Optional[int] = None
    temporal_pooling: str = "mean"  # mean/max/attention
    return_frame_embeddings: bool = False

class VisionSimilarityRequest(BaseModel):
    images1: Union[str, List[str]]
    images2: Union[str, List[str]]
    cross_modal: bool = False  # For CLIP text-image similarity
    texts: Optional[Union[str, List[str]]] = None  # If cross_modal
```

### 4.2 Response Schemas: `src/service/schemas/responses.py`

**New Classes:**

```python
class VisionEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    shape: Tuple[int, int]
    processing_time: float
    image_count: int

class VisionFrameEmbedResponse(BaseModel):
    aggregated_embedding: List[float]
    frame_embeddings: Optional[List[List[float]]]  # If requested
    frame_count: int
    shape: Tuple[int, int]
    processing_time: float

class VisionSimilarityResponse(BaseModel):
    similarity_matrix: List[List[float]]
    shape: Tuple[int, int]
    processing_time: float
```

---

## 5. Validation and Utilities

### 5.1 New Validation Module: `src/service/utils/vision_validation.py`

**Pattern:** Similar to `validation.py` (224 lines)

**Functions Needed:**

```python
def validate_base64_image(b64_string: str, field_name: str) -> None:
    """Validate base64 encoded image can be decoded."""

def validate_image_dimensions(img: PIL.Image, field_name: str) -> None:
    """Check image doesn't exceed max dimensions."""

def validate_image_size(b64_string: str, field_name: str) -> None:
    """Check base64 size is within limits."""

def validate_image_batch(images: List[str], field_name: str) -> None:
    """Validate batch of images against limits."""

def validate_frame_sequence(frames: List[str], field_name: str) -> None:
    """Validate video frame sequence."""
```

### 5.2 New Utility: `src/service/utils/image_processing.py`

**New File Needed:**

```python
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def decode_base64_image(b64_string: str) -> PIL.Image:
    """Decode base64 string to PIL Image."""

def preprocess_image(img: PIL.Image, target_size: int) -> PIL.Image:
    """Resize and prepare image for model input."""

def image_to_rgb(img: PIL.Image) -> PIL.Image:
    """Convert to RGB if needed (handles RGBA, L, etc.)."""

def batch_decode_images(b64_list: List[str]) -> List[PIL.Image]:
    """Decode multiple base64 images with error handling."""

def get_image_stats(img: PIL.Image) -> Dict[str, Any]:
    """Return image metadata (size, mode, format)."""
```

### 5.3 Update: `src/utils/convenience.py`

**Add Vision Helpers:**

```python
async def quick_vision_embed(
    images: Union[str, List[str]],
    model_name: str = 'openclip-vit-b-32',
    **kwargs
) -> np.ndarray:
    """Quick vision embedding generation."""

async def quick_frame_embed(
    frames: List[str],
    model_name: str = 'openclip-vit-b-32',
    temporal_pooling: str = 'mean',
    **kwargs
) -> np.ndarray:
    """Quick frame sequence embedding."""
```

---

## 6. Dependencies and Requirements

### 6.1 Update: `requirements.in`

**Current Vision-Related:**
```
pillow  # Already present!
```

**Add:**
```
# Vision models
open-clip-torch>=2.20.0  # OpenCLIP models
timm>=0.9.0  # For DINOv2 and ViT architectures

# Image processing
opencv-python>=4.8.0  # Advanced image ops (optional)
torchvision>=0.16.0  # Vision transforms

# Already have:
# torch
# transformers (may still be useful for CLIP)
# pillow
```

### 6.2 Model Downloads

**Storage Considerations:**
- OpenCLIP models: 300MB - 1.5GB depending on variant
- DINOv2 models: 80MB (small) - 1.1GB (large)
- Cache location: Same as text models (`transformers` cache)

**Docker Image Impact:**
- Current image: ~8GB with text models
- With vision models: Estimate +2-3GB (if caching common models)
- Consider multi-stage builds or model-on-demand loading

---

## 7. Testing Infrastructure

### 7.1 New Test Files

**Test Visual Engine:**
- `test_visual_engine.py` - Unit tests for VisualEngine
- `test_vision_routes.py` - Route endpoint tests
- `test_image_processing.py` - Image utility tests
- `test_vision_similarity.py` - Similarity computation tests

**Pattern from Existing:**
- `test_embedding_engine.py` - Reference for engine testing patterns
- `test_segmentation.py` - Adapt for multi-scale/frame testing
- `test_large_batch.py` - Test large image batches

### 7.2 Test Data

**Need Sample Images:**
```
tests/fixtures/images/
  ├── single_image.jpg
  ├── rgb_image.png
  ├── rgba_image.png
  ├── grayscale_image.jpg
  ├── high_res_image.jpg (4K)
  ├── low_res_image.jpg (64x64)
  └── frames/
      ├── frame_001.jpg
      ├── frame_002.jpg
      └── ...
```

### 7.3 Load Testing

**Update:** `working_dir/load_test.py`

```python
# Add vision endpoint tests
def test_vision_embeddings(url, num_requests, concurrency):
    # Load sample base64 images
    # POST to /vision/embeddings
    # Measure throughput

def test_frame_sequences(url, num_requests, concurrency):
    # Test video frame batches
    # Measure temporal pooling performance
```

---

## 8. Deployment and Infrastructure

### 8.1 Ray Serve Deployment

**File:** `src/service/ray_deployment.py`

**Options:**

**Option A: Unified Deployment (text + vision)**
```python
@serve.deployment(
    name="multimodal-service",
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 0.3,  # Increase GPU share for vision
    },
    max_ongoing_requests=150,  # Reduce due to larger models
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 12,
    },
)
```

**Option B: Separate Deployments**
```python
# Keep existing text deployment
@serve.deployment(name="embedding-service", ...)
class EmbeddingServiceDeployment:
    pass

# Add new vision deployment
@serve.deployment(
    name="vision-service",
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 0.5,  # More GPU for vision
    },
    max_ongoing_requests=100,
)
class VisionServiceDeployment:
    pass
```

**Recommendation:** Option A for code sharing, Option B for independent scaling.

### 8.2 Kubernetes Manifests

**File:** `k8s/rayservice-local.yaml`

**Changes Needed:**
- Increase GPU allocation per worker (0.5 → 1.0 full GPU if vision-only)
- Update resource limits for larger models
- Add vision-specific environment variables

```yaml
workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 2
    rayStartParams:
      num-gpus: '1'  # Full GPU per worker for vision
    template:
      spec:
        containers:
        - name: ray-worker
          resources:
            limits:
              nvidia.com/gpu: '1'
              memory: 32Gi  # Increase for large vision models
            requests:
              memory: 16Gi
```

### 8.3 Docker Updates

**No major changes needed** - Dockerfile already supports:
- CUDA via nvidia-container-toolkit
- Python 3.12
- Volume mounting for model cache
- Multi-stage builds possible for optimization

---

## 9. Documentation Updates

### 9.1 New Documentation Files

**Create:**
- `docs/VISION_API.md` - Vision endpoint documentation
- `docs/VISION_MODELS.md` - Supported models and usage
- `docs/IMAGE_FORMATS.md` - Input format specifications
- `VISION_SERVICE_MIGRATION_PLAN.md` - (this document)

### 9.2 Update Existing Docs

**Files to Update:**
- `README.md` - Add vision capabilities overview
- `MAKE_BUILD_REDEPLOY_TEST.md` - Add vision-specific test targets
- `docs/OPS_NOTES.md` - Vision model deployment notes

### 9.3 API Documentation

**FastAPI Auto-Docs:**
- Will auto-generate from Pydantic schemas
- Add detailed docstrings to vision routes
- Include example base64 images in OpenAPI examples

---

## 10. Makefile Targets

### 10.1 New Targets for Vision Testing

**Add to:** `Makefile`

```makefile
.PHONY: test-vision-local
test-vision-local:
	@echo "Testing vision endpoints locally..."
	python working_dir/vision_load_test.py --url http://localhost:8090

.PHONY: test-vision-ingress
test-vision-ingress:
	@echo "Testing vision endpoints via ingress..."
	python working_dir/vision_load_test.py --url $(INGRESS_URL)

.PHONY: benchmark-vision
benchmark-vision:
	@echo "Benchmarking vision models..."
	python tests/benchmark_vision.py
```

---

## 11. Migration Strategy

### 11.1 Phase 1: Core Engine (Week 1-2)

1. Create `visual_engine.py` with OpenCLIP support
2. Implement base64 decode and image preprocessing
3. Add single image embedding generation
4. Unit tests for engine functionality
5. Local testing without service layer

**Deliverable:** Working VisualEngine class that can embed images

### 11.2 Phase 2: Service Integration (Week 2-3)

1. Add vision routes to FastAPI app
2. Implement request/response schemas
3. Create visual_engine_manager
4. Add validation utilities
5. Integration tests for endpoints

**Deliverable:** Vision API endpoints accessible locally

### 11.3 Phase 3: Advanced Features (Week 3-4)

1. Frame sequence support (temporal pooling)
2. DINOv2 model integration
3. Multi-scale processing
4. Similarity search optimization
5. Performance tuning and batching

**Deliverable:** Full-featured vision service with video support

### 11.4 Phase 4: Deployment (Week 4-5)

1. Ray Serve deployment configuration
2. Kubernetes manifest updates
3. Load testing and benchmarking
4. Documentation finalization
5. Production deployment

**Deliverable:** Production-ready vision service

---

## 12. Code Reuse Opportunities

### 12.1 Keep As-Is (Minimal/No Changes)

✅ **Configuration System**
- `src/config_handler.py` - Base class works for both
- `src/config.py` - Just add `vision` section

✅ **Service Infrastructure**
- `src/service/main.py` - Just register new routes
- `src/service/core/dependencies.py` - Add vision semaphores
- `src/service/core/exceptions.py` - Reuse all exception classes
- `src/service/middleware/*` - All middleware reusable

✅ **Utilities**
- `src/service/utils/serialization.py` - numpy_to_list works for vision
- Most of validation patterns (adapt validators)

✅ **Deployment**
- `Dockerfile` - No changes needed
- `Makefile` - Add targets, keep existing
- `k8s/` manifests - Adjust resources only

### 12.2 Adapt/Extend

🔄 **Engine Pattern**
- Copy structure from `embedding_engine.py`
- Adapt: Input processing (text → images)
- Adapt: Pooling strategies (tokens → pixels/patches)
- Keep: Async/sync pattern, batching, memory management

🔄 **Manager Pattern**
- Copy structure from `engine_manager.py`
- Adapt: Model registry (text → vision models)
- Keep: Singleton, lazy loading, semaphores, cleanup

🔄 **Routes**
- Copy structure from `routes/embeddings.py`
- Adapt: Input validation (text → base64 images)
- Keep: Error handling, response formats, async patterns

### 12.3 New Components

🆕 **Image Processing**
- New module: `src/service/utils/image_processing.py`
- Base64 decode/encode
- PIL/OpenCV operations
- Format conversions

🆕 **Vision Models**
- New integrations: OpenCLIP, DINOv2
- Model wrappers for consistent interface
- Preprocessing pipelines per model

---

## 13. Risk Analysis

### 13.1 Technical Risks

**Risk 1: GPU Memory with Large Vision Models**
- **Impact:** High - ViT-L models need 2-4GB VRAM
- **Mitigation:**
  - Start with smaller models (ViT-B-32)
  - Implement batch size auto-tuning
  - Monitor GPU memory usage closely

**Risk 2: Image Size Variability**
- **Impact:** Medium - Real-world images vary greatly
- **Mitigation:**
  - Strict input validation
  - Resize to fixed resolution
  - Limit max resolution (4K cap)

**Risk 3: Base64 Encoding Overhead**
- **Impact:** Medium - 33% size increase + encode/decode CPU cost
- **Mitigation:**
  - Benchmark and document overhead
  - Consider binary upload option later
  - Gzip compression on transport

**Risk 4: Model Download Sizes**
- **Impact:** Low-Medium - First-run downloads can be slow
- **Mitigation:**
  - Pre-download models in Docker build
  - Document expected sizes
  - Model caching in persistent volume

### 13.2 Operational Risks

**Risk 1: Increased Deployment Size**
- **Current:** 8GB Docker image
- **With Vision:** 10-11GB estimated
- **Mitigation:** Multi-stage builds, separate images

**Risk 2: Different Scaling Characteristics**
- **Text:** High throughput, fast inference
- **Vision:** Lower throughput, slower inference
- **Mitigation:** Separate deployments or careful tuning

---

## 14. Performance Expectations

### 14.1 Throughput Estimates

**Text Service (Current):**
- ~1000 requests/min @ 50 concurrent
- Mean latency: 50-200ms
- P95 latency: <500ms

**Vision Service (Projected):**
- ~200-500 requests/min @ 20 concurrent (OpenCLIP ViT-B-32)
- Mean latency: 100-300ms per image
- P95 latency: <1000ms
- Batch of 10 images: 300-800ms

**Factors:**
- Image size (larger = slower preprocessing)
- Model size (ViT-L vs ViT-B)
- Batch size (GPU saturation point)
- Frame sequences (temporal dimension)

### 14.2 Resource Requirements

**Per Replica:**
- CPU: 2 cores (image decode/preprocessing)
- GPU: 0.5-1.0 full GPU
- RAM: 8-16GB (model + image buffers)
- Disk: 2-3GB (model cache)

**Cluster:**
- 2 worker nodes with RTX A4500 (20GB VRAM each)
- Can run 2-4 vision replicas total
- Or mix: 2 vision + 2 text replicas per GPU

---

## 15. Alternative Approaches

### 15.1 Separate Repository vs. Unified Codebase

**Option A: Fork/New Repo (VisionService)**
- ✅ Clean separation of concerns
- ✅ Independent versioning
- ✅ Simpler codebase
- ❌ Code duplication (service layer, deployment)
- ❌ Two repos to maintain

**Option B: Unified Multimodal Service**
- ✅ Code reuse (middleware, deployment, config)
- ✅ Single deployment unit
- ✅ Easier cross-modal features (CLIP text-image)
- ❌ Larger codebase
- ❌ Coupled scaling

**Recommendation:** Unified codebase for code reuse benefits.

### 15.2 Model Serving Approaches

**Option A: Load All Models on Startup**
- ✅ Fast inference (no loading delay)
- ❌ High memory usage
- ❌ Inflexible

**Option B: Lazy Loading (Current Pattern)**
- ✅ Memory efficient
- ✅ Flexible model switching
- ❌ First-request latency spike

**Option C: Model Pool with Pre-warming**
- ✅ Balance of speed and memory
- ✅ Common models pre-loaded
- ❌ More complex

**Recommendation:** Lazy loading (Option B) - matches existing pattern.

### 15.3 Input Format Options

**Option A: Base64 Encoded (Proposed)**
- ✅ Simple REST API
- ✅ No file upload infrastructure
- ❌ 33% size overhead
- ❌ Encode/decode CPU cost

**Option B: Multipart Form Upload**
- ✅ No encoding overhead
- ✅ Browser-friendly
- ❌ More complex API
- ❌ Harder to test

**Option C: URL References**
- ✅ No payload size
- ❌ Requires image fetching
- ❌ Security concerns

**Recommendation:** Base64 (Option A) - consistent with text input simplicity.

---

## 16. Open Questions

### 16.1 Architecture Decisions Needed

1. **Deployment Strategy:** Unified service or separate vision deployment?
2. **Model Selection:** Which OpenCLIP variants to support initially?
3. **Frame Aggregation:** Mean pooling sufficient or add attention mechanisms?
4. **Cross-Modal:** Should CLIP text-image similarity be supported?
5. **Scaling:** Same Ray Serve autoscaling params or vision-specific?

### 16.2 Feature Prioritization

**Must Have (MVP):**
- Single image embedding (OpenCLIP)
- Batch image embedding
- Image-to-image similarity
- Base64 input/output

**Should Have (V2):**
- Frame sequence support
- DINOv2 models
- Temporal pooling options
- Multi-scale processing

**Nice to Have (Future):**
- CLIP text-image cross-modal
- Video file upload
- Real-time streaming
- Model fine-tuning API

### 16.3 Integration Points

1. **Text + Vision Search:** Should search support text query → image results?
2. **Multimodal Embeddings:** Unified embedding space (CLIP) or separate?
3. **Storage:** Should we cache computed embeddings?
4. **Monitoring:** Same Prometheus metrics or vision-specific?

---

## 17. Next Steps

### 17.1 Immediate Actions (Before Coding)

1. ✅ **This Document** - Review and approve migration plan
2. ⏳ **Model Testing** - Benchmark OpenCLIP and DINOv2 on target GPUs
3. ⏳ **Dependency Review** - Test `open-clip-torch` and `timm` installation
4. ⏳ **Input Format POC** - Validate base64 encode/decode performance
5. ⏳ **Architecture Decision** - Unified vs separate deployment

### 17.2 Development Setup

1. Create feature branch: `feature/vision-engine`
2. Set up development environment with vision dependencies
3. Download and cache sample models
4. Prepare test image dataset
5. Review OpenCLIP and DINOv2 documentation

### 17.3 Proof of Concept

**Week 1 Goal:** Standalone VisualEngine prototype

```python
# Minimal working example
from src.components.visual_engine import VisualEngine

engine = VisualEngine(model_name='openclip-vit-b-32')
b64_images = ["iVBORw0KGgoAAAANS...", ...]  # Sample base64
embeddings = await engine.embed(b64_images)
print(embeddings.shape)  # (2, 512)
```

---

## 18. Success Criteria

### 18.1 Functional Requirements

- ✅ Single image embedding with OpenCLIP
- ✅ Batch image embedding (1-100 images)
- ✅ Frame sequence aggregation (mean/max pooling)
- ✅ Image-to-image similarity computation
- ✅ Base64 input format support
- ✅ DINOv2 model support
- ✅ GPU acceleration with memory management

### 18.2 Performance Requirements

- ⚡ Single image latency: <200ms (ViT-B-32)
- ⚡ Batch of 10 images: <800ms
- ⚡ Throughput: >200 req/min with 2 replicas
- ⚡ GPU memory: <10GB per replica
- ⚡ P95 latency: <1000ms

### 18.3 Quality Requirements

- 🧪 Unit test coverage: >80%
- 🧪 Integration tests for all endpoints
- 🧪 Load tests showing stable performance
- 📚 Complete API documentation
- 📚 Model usage guides
- 🚀 Production deployment successful

---

## 19. Conclusion

The existing EmbeddingService architecture is **well-suited for vision adaptation** with minimal structural changes. The async/sync pattern, manager-based model handling, Ray Serve deployment, and FastAPI service layer can all be reused with domain-specific adaptations.

**Key Success Factors:**
1. Follow established patterns from text service
2. Leverage existing infrastructure (Ray, K8s, Docker)
3. Maintain code quality and testing standards
4. Document vision-specific considerations
5. Plan for separate scaling characteristics

**Estimated Effort:**
- Core engine: 1-2 weeks
- Service integration: 1-2 weeks
- Advanced features: 1-2 weeks
- Deployment & testing: 1 week
- **Total: 4-6 weeks** for production-ready vision service

**Recommendation:** Proceed with unified codebase approach, implementing vision capabilities alongside existing text embeddings. This maximizes code reuse while enabling future multimodal features.

---

**Document Version:** 1.0
**Date:** October 24, 2025
**Status:** DRAFT - Awaiting Review
