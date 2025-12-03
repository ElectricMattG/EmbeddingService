# Vision Service Additional Capabilities from DINOv3

## Document Purpose

This document identifies **additional vision capabilities** beyond basic embeddings that should be exposed as service routes, based on the official DINOv3 documentation and examples. These are pre-trained capabilities that require no training - just inference.

---

## 1. Overview of Additional Capabilities

Based on DINOv3 documentation, we can expose:

1. **Semantic Segmentation** - Pixel-level classification (ADE20K trained)
2. **Dense Correspondence** - Patch matching between images
3. **PCA Visualization** - Rainbow object part visualization
4. **Video Segmentation Tracking** - Propagate masks across frames
5. **Depth Estimation** - Monocular depth prediction (SYNTHMIX trained)
6. **Object Detection** - Bounding boxes (COCO trained)
7. **Patch-level Features** - Get intermediate layer outputs with spatial structure

**Key Insight:** DINOv3 provides pre-trained "heads" (adapters) for specific tasks that we can load and use for inference without any training.

---

## 2. Semantic Segmentation

### 2.1 Capability Description

**What it does:** Per-pixel semantic segmentation using pre-trained M2F head on ADE20K dataset (150 classes).

**Input:** Single RGB image (base64 encoded)
**Output:** Segmentation map (H×W) with class IDs + confidence scores per pixel

**Pre-trained Model:**
- Backbone: ViT-7B/16 or smaller variants
- Head: M2F segmentor trained on ADE20K
- 150 output classes (indoor/outdoor scenes)

### 2.2 API Route Design

```python
POST /vision/segment

Request:
{
    "image": "base64_encoded_image",
    "return_probabilities": false,  // Return per-pixel class probabilities
    "inference_mode": "slide",      // "slide" or "whole"
    "input_size": 896,              // Processing resolution
    "output_activation": "softmax"  // Post-processing
}

Response:
{
    "segmentation_map": [[class_id, ...], ...],  // 2D array H×W
    "class_probabilities": [...],  // Optional: [150, H, W] if requested
    "class_names": ["wall", "floor", ...],  // ADE20K class names
    "shape": [H, W],
    "processing_time": 0.456
}
```

### 2.3 Implementation Notes

**Model Loading:**
```python
# Load segmentor with backbone
segmentor = torch.hub.load(
    REPO_DIR,
    'dinov3_vit7b16_ms',  # or smaller variants
    source="local",
    weights=<SEGMENTOR_CHECKPOINT_URL>,
    backbone_weights=<BACKBONE_CHECKPOINT_URL>
)
```

**Inference Pattern:**
```python
from dinov3.eval.segmentation.inference import make_inference
from functools import partial

# Raw predictions
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = segmentor(batch_img)

# Actual segmentation map with sliding window
segmentation_map = make_inference(
    batch_img,
    segmentor,
    inference_mode="slide",
    decoder_head_type="m2f",
    rescale_to=(img.size[-1], img.size[-2]),
    n_output_channels=150,
    crop_size=(896, 896),
    stride=(896, 896),
    output_activation=partial(torch.nn.functional.softmax, dim=1),
).argmax(dim=1, keepdim=True)
```

**Classes:** ADE20K 150 classes (wall, building, sky, floor, tree, ceiling, road, bed, window, grass, etc.)

---

## 3. Dense Correspondence / Patch Matching

### 3.1 Capability Description

**What it does:** Find corresponding patches between two images using cosine similarity of patch features.

**Use Cases:**
- Image alignment
- Object tracking
- 3D reconstruction preparation
- Visual SLAM

**Input:** Two RGB images
**Output:** Correspondence pairs (patch coordinates in image1 ↔ image2)

### 3.2 API Route Design

```python
POST /vision/correspondence

Request:
{
    "image1": "base64_encoded_image",
    "image2": "base64_encoded_image",
    "visualization_type": "sparse",  // "dense" or "sparse"
    "topk": 5,  // Top-k matches per patch
    "mask_threshold": 0.5,  // Foreground mask threshold
    "stratify_distance": 100.0  // Spatial separation for sparse matches
}

Response:
{
    "correspondences": [
        {
            "image1_location": [row, col],  // Patch center in pixels
            "image2_location": [row, col],
            "similarity": 0.95,
            "color": [r, g, b]  // PCA-based color for visualization
        },
        ...
    ],
    "dense_similarity_map": [...],  // Optional: full similarity heatmap
    "num_matches": 55,
    "processing_time": 0.234
}
```

### 3.3 Implementation Notes

**Extract Features:**
```python
# Get patch features for both images
with torch.inference_mode():
    feats1 = model.get_intermediate_layers(
        img1, n=range(n_layers), reshape=True, norm=True
    )[-1].squeeze().detach().cpu()  # [D, h, w]

    feats2 = model.get_intermediate_layers(
        img2, n=range(n_layers), reshape=True, norm=True
    )[-1].squeeze().detach().cpu()  # [D, h, w]
```

**Compute Correspondence:**
```python
# Normalize features
feats1 = F.normalize(feats1, p=2, dim=0)
feats2 = F.normalize(feats2, p=2, dim=0)

# Compute similarity matrix (all patches in img1 vs all in img2)
heatmaps = torch.einsum(
    "k f, f h w -> k h w",
    feats1.view(dim, -1).permute(1, 0),  # [h1*w1, D]
    feats2,  # [D, h2, w2]
)  # [h1*w1, h2, w2]

# Find best matches
patch_indices_img2 = heatmaps.flatten(start_dim=-2).argmax(dim=-1)
```

**PCA Color Encoding:**
```python
from sklearn.decomposition import PCA

# Fit PCA on foreground patches
pca = PCA(n_components=3, whiten=True)
pca.fit(fg_patches)

# Project features to RGB space
projected = pca.transform(features)
colors = torch.nn.functional.sigmoid(
    torch.from_numpy(projected).mul(2.0)
).numpy()
```

---

## 4. PCA Visualization (Rainbow Features)

### 4.1 Capability Description

**What it does:** Visualize object parts using PCA on patch features - the iconic "rainbow" visualizations from DINOv3 paper.

**Input:** Single image + optional foreground mask
**Output:** RGB image where colors represent semantic similarity (similar parts = similar colors)

### 4.2 API Route Design

```python
POST /vision/pca-visualization

Request:
{
    "image": "base64_encoded_image",
    "mask": "base64_encoded_mask",  // Optional: focus PCA on foreground
    "num_components": 3,  // Always 3 for RGB
    "apply_sigmoid": true  // Enhance color vibrancy
}

Response:
{
    "visualization": "base64_encoded_rgb_image",  // PCA-colored image
    "explained_variance": [0.45, 0.32, 0.15],  // Per-component
    "shape": [H, W, 3],
    "processing_time": 0.123
}
```

### 4.3 Implementation Notes

**Extract and Process:**
```python
# Get patch features
with torch.inference_mode():
    feats = model.get_intermediate_layers(
        img, n=range(n_layers), reshape=True, norm=True
    )[-1].squeeze().detach().cpu()  # [D, h, w]

# Fit PCA on foreground patches only
pca = PCA(n_components=3, whiten=True)
fg_patches = features[foreground_mask]
pca.fit(fg_patches)

# Project all patches
projected = pca.transform(features)  # [N, 3]
projected = projected.reshape(h, w, 3)

# Enhance colors with sigmoid
rgb = torch.nn.functional.sigmoid(
    torch.from_numpy(projected).mul(2.0)
).permute(2, 0, 1)  # [3, h, w]

# Apply foreground mask
rgb *= (fg_mask > 0.5)[None, :, :]
```

**Use Cases:**
- Object part discovery
- Visual debugging
- Understanding what the model "sees"
- Marketing/demo visualizations

---

## 5. Video Segmentation Tracking

### 5.1 Capability Description

**What it does:** Propagate segmentation masks across video frames using feature correspondence (non-parametric tracking).

**Input:**
- Video frames (sequence of base64 images)
- Segmentation mask for first frame

**Output:**
- Segmentation masks for all frames
- Optional: per-frame confidence scores

### 5.2 API Route Design

```python
POST /vision/track-segmentation

Request:
{
    "frames": ["base64_frame1", "base64_frame2", ...],  // Max 300 frames
    "first_frame_mask": "base64_encoded_mask",  // Instance segmentation
    "max_context_length": 7,  // Temporal window
    "neighborhood_size": 12,  // Spatial search radius
    "topk": 5,  // Top-k similar patches
    "temperature": 0.2,  // Softmax temperature
    "return_probabilities": false  // Return confidence per pixel
}

Response:
{
    "masks": ["base64_mask1", "base64_mask2", ...],  // Per-frame masks
    "probabilities": [...],  // Optional: [T, M, H, W] if requested
    "num_instances": 6,  // Number of tracked objects
    "processing_time": 5.678
}
```

### 5.3 Implementation Notes

**Core Algorithm:**
```python
def propagate(
    current_features: Tensor,  # [h, w, D]
    context_features: Tensor,  # [t, h, w, D]
    context_probs: Tensor,  # [t, h, w, M]
    neighborhood_mask: Tensor,  # [h, w, h, w]
    topk: int,
    temperature: float,
) -> Tensor:
    # Compute similarity current -> context
    dot = torch.einsum(
        "ijd, tuvd -> ijtuv",
        current_features,
        context_features,
    )  # [h, w, t, h, w]

    # Restrict to local neighborhood
    dot = torch.where(neighborhood_mask[:, :, None, :, :], dot, -torch.inf)

    # Select top-k
    k_th_largest = torch.topk(dot.flatten(2, -1).flatten(0, 1), dim=1, k=topk).values
    dot = torch.where(dot.flatten(2, -1).flatten(0, 1) >= k_th_largest[:, -1:], dot, -torch.inf)

    # Propagate via weighted average
    weights = F.softmax(dot / temperature, dim=1)
    current_probs = torch.mm(weights, context_probs.flatten(0, 2))

    return current_probs.unflatten(0, (h, w))
```

**Context Queue Management:**
```python
features_queue: list[Tensor] = []
probs_queue: list[Tensor] = []

for frame_idx in range(1, num_frames):
    current_feats = extract_features(frame)

    context_feats = torch.stack([first_feats, *features_queue], dim=0)
    context_probs = torch.stack([first_probs, *probs_queue], dim=0)

    current_probs = propagate(current_feats, context_feats, context_probs, ...)

    features_queue.append(current_feats)
    probs_queue.append(current_probs)
    if len(features_queue) > max_context_length:
        features_queue.pop(0)
        probs_queue.pop(0)
```

---

## 6. Depth Estimation

### 6.1 Capability Description

**What it does:** Monocular depth prediction using pre-trained depther head on SYNTHMIX dataset.

**Input:** Single RGB image
**Output:** Dense depth map (inverse depth or metric depth)

### 6.2 API Route Design

```python
POST /vision/depth

Request:
{
    "image": "base64_encoded_image",
    "input_size": 1024,  // Processing resolution
    "colormap": "spectral",  // Visualization colormap
    "return_raw_depth": true  // Return raw depth values
}

Response:
{
    "depth_map": [[depth_value, ...], ...],  // 2D array H×W
    "depth_visualization": "base64_encoded_colormap",  // Optional visualization
    "depth_range": [min_depth, max_depth],
    "shape": [H, W],
    "processing_time": 0.567
}
```

### 6.3 Implementation Notes

**Model Loading:**
```python
depther = torch.hub.load(
    REPO_DIR,
    'dinov3_vit7b16_dd',
    source="local",
    weights=<DEPTHER_CHECKPOINT_URL>,
    backbone_weights=<BACKBONE_CHECKPOINT_URL>
)
```

**Inference:**
```python
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        depths = depther(batch_img)  # [B, 1, H, W]
```

**Visualization:**
```python
from matplotlib import colormaps

# Normalize and apply colormap
depth_viz = colormaps["Spectral"](depth_normalized)
```

**Evaluation Metrics (if ground truth available):**
- Absolute Relative Error (AbsRel)
- δ < 1.25, δ < 1.25², δ < 1.25³ (accuracy thresholds)

---

## 7. Object Detection

### 7.1 Capability Description

**What it does:** Bounding box detection using pre-trained detector head on COCO dataset.

**Input:** Single RGB image
**Output:** Bounding boxes + class labels + confidence scores

### 7.2 API Route Design

```python
POST /vision/detect

Request:
{
    "image": "base64_encoded_image",
    "confidence_threshold": 0.5,  // Minimum detection confidence
    "nms_threshold": 0.4,  // Non-maximum suppression
    "max_detections": 100  // Maximum boxes to return
}

Response:
{
    "detections": [
        {
            "bbox": [x, y, width, height],  // COCO format
            "class_id": 1,
            "class_name": "person",
            "confidence": 0.95
        },
        ...
    ],
    "num_detections": 15,
    "processing_time": 0.234
}
```

### 7.3 Implementation Notes

**Model Loading:**
```python
detector = torch.hub.load(
    REPO_DIR,
    'dinov3_vit7b16_de',
    source="local",
    weights=<DETECTOR_CHECKPOINT_URL>,
    backbone_weights=<BACKBONE_CHECKPOINT_URL>
)
```

**Classes:** COCO 80 object classes

---

## 8. Zero-Shot Classification with dino.txt

### 8.1 Capability Description

**What it does:** Zero-shot image classification using text-aligned features (CLIP-like capability).

**Input:** Image + text prompts
**Output:** Classification scores per prompt

### 8.2 API Route Design

```python
POST /vision/classify-zeroshot

Request:
{
    "image": "base64_encoded_image",
    "texts": ["a photo of a dog", "a photo of a cat", "a photo of a car"],
    "templates": "imagenet",  // Use ImageNet prompt templates
    "top_k": 5  // Return top-k predictions
}

Response:
{
    "predictions": [
        {
            "text": "a photo of a dog",
            "score": 0.87,
            "rank": 1
        },
        ...
    ],
    "processing_time": 0.123
}
```

### 8.3 Implementation Notes

**Model Loading:**
```python
model, tokenizer = torch.hub.load(
    REPO_DIR,
    'dinov3_vitl16_dinotxt_tet1280d20h24l',
    weights=<CHECKPOINT_URL>,
    backbone_weights=<BACKBONE_CHECKPOINT_URL>
)
```

**Inference:**
```python
# Encode image
image_features = model.encode_image(image_tensor)
image_features /= image_features.norm(dim=-1, keepdim=True)

# Encode texts
tokenized_texts = tokenizer.tokenize(texts).cuda()
text_features = model.encode_text(tokenized_texts)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute similarity
similarity = text_features @ image_features.T
```

**Template Augmentation:**
```python
openai_imagenet_templates = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    # ... 80 templates total
]

# Average over all templates
for classname in classnames:
    texts = [template(classname) for template in templates]
    text_embeddings = encode_text(texts)
    class_embedding = text_embeddings.mean(dim=0)
```

---

## 9. Patch-Level Feature Extraction

### 9.1 Capability Description

**What it does:** Extract intermediate layer patch features with spatial structure preserved.

**Use Cases:**
- Custom downstream tasks
- Feature analysis
- Fine-grained retrieval
- Attention visualization

### 9.2 API Route Design

```python
POST /vision/patch-features

Request:
{
    "image": "base64_encoded_image",
    "layers": [11, 23],  // Which transformer layers (0-indexed)
    "reshape": true,  // Return spatial structure [D, h, w]
    "normalize": true,  // L2 normalize features
    "return_cls_token": false  // Include CLS token separately
}

Response:
{
    "patch_features": [...],  // [L, D, h, w] or [L, h*w, D]
    "cls_token": [...],  // Optional: [L, D]
    "patch_size": 16,
    "feature_dim": 1024,
    "spatial_shape": [h, w],
    "processing_time": 0.089
}
```

### 9.3 Implementation Notes

**Extract Multi-Layer Features:**
```python
with torch.inference_mode():
    feats = model.get_intermediate_layers(
        img,
        n=[11, 23],  # Layer indices
        reshape=True,  # [B, D, h, w] vs [B, N, D]
        norm=True  # L2 normalize
    )
# Returns list of tensors, one per layer
```

**With CLS Token:**
```python
outputs = model.forward_features(img)
cls_token = outputs['x_norm_clstoken']  # [B, D]
patch_tokens = outputs['x_norm_patchtokens']  # [B, N, D]
```

---

## 10. Implementation Priority & Complexity

### 10.1 Effort Matrix

| Capability | Priority | Complexity | Dependencies | Estimated Effort |
|------------|----------|------------|--------------|------------------|
| **Patch Features** | High | Low | Base model only | 1-2 days |
| **Dense Correspondence** | High | Medium | Base model + PCA | 3-4 days |
| **PCA Visualization** | Medium | Low | Base model + sklearn | 2-3 days |
| **Segmentation** | High | Medium | Segmentor head + inference utils | 4-5 days |
| **Zero-Shot Classification** | High | Medium | dino.txt model + tokenizer | 3-4 days |
| **Depth Estimation** | Medium | Medium | Depther head | 3-4 days |
| **Object Detection** | Medium | Medium | Detector head | 3-4 days |
| **Video Tracking** | Low | High | Complex state management | 7-10 days |

### 10.2 Quick Wins (Week 1-2)

1. **Patch Features** - Straightforward extension of embedding extraction
2. **PCA Visualization** - Useful for demos, low complexity
3. **Dense Correspondence** - High value, moderate complexity

### 10.3 High-Value Features (Week 3-4)

4. **Segmentation** - Very useful, needs M2F head
5. **Zero-Shot Classification** - Competitive with CLIP
6. **Depth Estimation** - Differentiator vs CLIP/OpenCLIP

### 10.4 Advanced Features (Week 5+)

7. **Object Detection** - Standard CV task, lower priority
8. **Video Tracking** - Complex, specialized use case

---

## 11. Service Architecture Updates

### 11.1 Manager Extensions

```python
class VisualEngineManager:
    """Extended to support task-specific heads."""

    def __init__(self):
        self._backbone: Optional[VisualEngine] = None
        self._heads: Dict[str, nn.Module] = {
            'segmentor': None,
            'depther': None,
            'detector': None,
            'dinotxt': None,
        }

    async def load_head(self, task: str, model_name: str):
        """Load task-specific head with backbone."""
        if task == 'segmentation':
            head = torch.hub.load(..., 'dinov3_vit7b16_ms', ...)
        elif task == 'depth':
            head = torch.hub.load(..., 'dinov3_vit7b16_dd', ...)
        # etc.
        self._heads[task] = head

    async def segment(self, image: str, **kwargs):
        """Run segmentation on image."""
        head = await self._ensure_head_loaded('segmentor')
        return await self._run_segmentation(image, head, **kwargs)
```

### 11.2 Route Organization

```
src/service/routes/
├── vision_embeddings.py       # Basic embeddings
├── vision_segmentation.py     # Semantic segmentation
├── vision_correspondence.py   # Dense/sparse matching
├── vision_visualization.py    # PCA rainbow viz
├── vision_depth.py            # Depth estimation
├── vision_detection.py        # Object detection
├── vision_tracking.py         # Video tracking
└── vision_classification.py   # Zero-shot with text
```

### 11.3 Configuration Updates

```python
"vision": {
    # ... existing basic config ...

    # Task-specific settings
    "segmentation": {
        "default_model": "dinov3_vit7b16_ms",
        "num_classes": 150,  # ADE20K
        "inference_mode": "slide",
        "crop_size": 896,
    },
    "depth": {
        "default_model": "dinov3_vit7b16_dd",
        "input_size": 1024,
        "colormap": "spectral",
    },
    "detection": {
        "default_model": "dinov3_vit7b16_de",
        "num_classes": 80,  # COCO
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
    },
    "correspondence": {
        "topk": 5,
        "temperature": 0.2,
        "neighborhood_size": 12,
    },
    "tracking": {
        "max_context_length": 7,
        "max_frames": 300,
    }
}
```

---

## 12. Resource Requirements

### 12.1 Model Weights Storage

| Model | Size | Task |
|-------|------|------|
| ViT-7B backbone | ~26GB | Base features |
| Segmentor (M2F) | ~100MB | Semantic segmentation |
| Depther | ~100MB | Depth estimation |
| Detector | ~100MB | Object detection |
| dino.txt | ~300MB | Zero-shot classification |
| **Total** | **~27GB** | All tasks |

**Docker Image Impact:** +27GB if all heads pre-cached

**Alternative:** Download on-demand (first use adds ~30s latency)

### 12.2 GPU Memory

| Task | VRAM (single image) | Notes |
|------|---------------------|-------|
| Embeddings | 2-4GB | Baseline |
| Segmentation | 4-6GB | Sliding window |
| Depth | 3-5GB | Full resolution |
| Detection | 3-5GB | COCO heads |
| Correspondence | 5-8GB | Similarity matrices |
| Video Tracking | 8-12GB | Context queue |

**Recommendation:**
- 16GB VRAM minimum for all tasks
- 24GB for video tracking
- RTX A4500 (20GB) works well

### 12.3 Dependencies

**Additional Python Packages:**
```txt
# Already have: torch, torchvision, timm

# New requirements:
scikit-learn>=1.3.0  # PCA for visualization
scipy>=1.11.0        # Signal processing for tracking
matplotlib>=3.7.0    # Colormaps for depth/segmentation
mediapy>=1.1.0       # Video handling (optional)
```

---

## 13. Testing Strategy

### 13.1 Unit Tests Per Capability

```python
# tests/test_vision_segmentation.py
@pytest.mark.asyncio
async def test_semantic_segmentation(visual_engine, sample_image):
    result = await visual_engine.segment(
        sample_image,
        return_probabilities=True
    )
    assert result["segmentation_map"].shape == (H, W)
    assert len(result["class_probabilities"]) == 150

# tests/test_vision_correspondence.py
@pytest.mark.asyncio
async def test_dense_correspondence(visual_engine, sample_images):
    result = await visual_engine.find_correspondences(
        sample_images[0],
        sample_images[1],
        visualization_type="sparse"
    )
    assert len(result["correspondences"]) > 0
    assert all(0 <= c["similarity"] <= 1 for c in result["correspondences"])
```

### 13.2 Integration Tests

```python
def test_segmentation_endpoint(client, sample_image):
    response = client.post(
        "/vision/segment",
        json={"image": sample_image, "return_probabilities": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "segmentation_map" in data
    assert "class_names" in data
```

---

## 14. Documentation Requirements

### 14.1 Per-Task Documentation

Each capability needs:
1. **Task description** - What it does, use cases
2. **Model info** - Backbone, head, training dataset
3. **API reference** - Request/response schemas with examples
4. **Performance** - Latency, throughput, memory
5. **Limitations** - Known failure cases
6. **Examples** - curl, Python, JavaScript

### 14.2 Visual Documentation

- **Segmentation:** Example input/output with class overlays
- **Correspondence:** Side-by-side with matching lines
- **PCA:** Rainbow visualizations with explanations
- **Depth:** RGB + colorized depth maps
- **Detection:** Images with bounding boxes

---

## 15. Phased Rollout Plan

### Phase 1: Core Extensions (Week 1-2)
- ✅ Patch-level features
- ✅ PCA visualization
- ✅ Dense correspondence

**Deliverable:** Extended vision API with feature extraction variants

### Phase 2: Task-Specific Heads (Week 3-4)
- ✅ Semantic segmentation
- ✅ Zero-shot classification (dino.txt)

**Deliverable:** Multi-task vision service

### Phase 3: Advanced Tasks (Week 5-6)
- ✅ Depth estimation
- ✅ Object detection

**Deliverable:** Comprehensive CV service

### Phase 4: Video & Optimization (Week 7+)
- ✅ Video segmentation tracking
- ✅ Performance optimization
- ✅ Model caching strategies

**Deliverable:** Production-ready with all features

---

## 16. Competitive Analysis

### 16.1 vs. OpenCLIP

**DINOv3 Advantages:**
- ✅ Semantic segmentation (ADE20K)
- ✅ Dense correspondence without fine-tuning
- ✅ Video tracking
- ✅ Depth estimation
- ✅ PCA-based object part discovery

**OpenCLIP Advantages:**
- ✅ Native text-image alignment
- ✅ Faster inference (smaller models)
- ✅ Better zero-shot on some benchmarks

**Recommendation:** Support both! OpenCLIP for embeddings/similarity, DINOv3 for advanced vision tasks.

### 16.2 vs. CLIP + SAM

**DINOv3 Unified Approach:**
- Single model for embeddings, segmentation, depth, correspondence
- No need for separate SAM deployment
- Coherent feature space across tasks

**Trade-off:**
- Larger model size
- Slower inference per task
- More VRAM required

---

## 17. Business Value

### 17.1 Use Cases Enabled

**Semantic Segmentation:**
- Scene understanding for robotics
- Content moderation (detect inappropriate content regions)
- Medical imaging analysis
- Autonomous vehicle perception

**Dense Correspondence:**
- 3D reconstruction from images
- Image alignment for panoramas
- Visual SLAM
- Augmented reality applications

**Video Tracking:**
- Surveillance and monitoring
- Sports analytics
- Video editing (object removal/replacement)
- Action recognition

**Depth Estimation:**
- 3D scene understanding
- Obstacle detection
- AR/VR applications
- Photography effects (portrait mode)

### 17.2 Market Differentiation

**Current Market (CLIP-based services):**
- Focus: Embeddings + similarity + zero-shot classification
- Gap: Advanced vision tasks require separate models/services

**Our Differentiation:**
- Unified vision service with 8+ capabilities
- Single API for embeddings → segmentation → depth → tracking
- Pre-trained on high-quality datasets (ADE20K, COCO, SYNTHMIX)
- No fine-tuning required

**Target Customers:**
- Robotics companies
- AR/VR developers
- Video analytics platforms
- Computer vision researchers
- Autonomous systems builders

---

## 18. Risk Mitigation

### 18.1 Technical Risks

**Risk:** Model size explosion (27GB+ total)
**Mitigation:**
- Lazy loading of task-specific heads
- Separate deployment option per task
- Model compression research

**Risk:** VRAM exhaustion with multiple tasks
**Mitigation:**
- Task queuing system
- Per-task VRAM limits
- Dynamic unloading of unused heads

**Risk:** Slow first-request latency (model downloads)
**Mitigation:**
- Pre-warm common models in Docker
- Async model loading with progress tracking
- Cache models in persistent volume

### 18.2 Operational Risks

**Risk:** API complexity (8+ endpoints × parameters)
**Mitigation:**
- Comprehensive OpenAPI documentation
- SDK generation for Python/JS/Go
- Interactive playground/demo

**Risk:** Different scaling needs per task
**Mitigation:**
- Separate Ray Serve deployments per task family
- Per-task autoscaling configuration
- Load-based routing

---

## 19. Success Metrics

### 19.1 Technical Metrics

- **Latency P95 < 1s** for single-image tasks
- **Throughput > 100 req/min** per replica
- **VRAM usage < 16GB** per replica for standard tasks
- **Model load time < 30s** for any head

### 19.2 Quality Metrics

**Segmentation:** mIoU on ADE20K validation
**Depth:** AbsRel error on NYUv2
**Detection:** mAP on COCO val
**Correspondence:** PCK@0.1 on standard benchmarks

---

## 20. Conclusion

DINOv3 provides **8 additional capabilities** beyond basic embeddings:

**Quick Wins (High Priority):**
1. Patch-level features
2. Dense correspondence
3. PCA visualization

**High Value (Medium Priority):**
4. Semantic segmentation
5. Zero-shot classification

**Advanced (Lower Priority):**
6. Depth estimation
7. Object detection
8. Video tracking

**Recommendation:**
- **Phase 1:** Implement top 3 quick wins alongside basic embeddings
- **Phase 2:** Add segmentation + zero-shot (strong differentiators)
- **Phase 3:** Depth + detection if customer demand exists
- **Phase 4:** Video tracking for specialized use cases

**Total Effort:** 6-8 weeks for full implementation
**Minimum Viable Product:** 2-3 weeks (embeddings + patch features + correspondence)

**Key Decision:** Unified service vs task-specific deployments?
→ Start unified, split if scaling/VRAM becomes issue.

---

**Document Version:** 1.0
**Date:** October 24, 2025
**Status:** Planning - Awaiting Review
