# EmbeddingService

A production-ready, GPU-accelerated text embedding service built with FastAPI, Ray Serve, and deployed on Kubernetes. Provides high-performance semantic embeddings using transformer models with enterprise-grade features including authentication, SSL, monitoring, and scalability.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Deployment Environment](#deployment-environment)
- [API Endpoints](#api-endpoints)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Development](#development)
- [Monitoring & Health](#monitoring--health)
- [Performance](#performance)

---

## Overview

EmbeddingService transforms text into dense vector representations (embeddings) for semantic similarity, search, clustering, and AI applications. It leverages GPU acceleration for high throughput and uses Ray Serve for horizontal scaling.

### Key Capabilities

- Generate embeddings for single texts or batches (up to 1,024 texts)
- Automatic text segmentation for documents longer than 512 tokens
- Cosine similarity computation between text pairs
- Multiple pre-trained models (MiniLM, BERT variants, Legal-BERT)
- GPU-accelerated inference with T4/A4500 GPUs
- Production-ready with SSL, API key authentication, and monitoring

---

## Architecture

### System Architecture

The service is deployed on Kubernetes with the following components:

**DNS Layer:**
- `embeddings.192-168-1-3.nip.io` → API Service
- `dashboard.192-168-1-3.nip.io` → Ray Dashboard

**Ingress Layer (NGINX Ingress Controller):**
- SSL Termination (Let's Encrypt certificates)
- API Key Authentication for embeddings endpoint
- Basic Authentication for dashboard
- Load balancing with hostNetwork mode

**Kubernetes Cluster:**
- **RayService**: `embedding-service`
  - ClusterIP Service: `embedding-service-serve-svc:8000`

- **Ray Head Pod** (System/Control Plane):
  - Ray Dashboard on port 8265
  - GCS Server on port 6379
  - FastAPI App on port 8000
  - Resources: 4 CPU, 8Gi RAM, no GPU

- **Ray Worker Pods** (GPU Inference):
  - NVIDIA A4500 GPUs (2x 21GB VRAM each)
  - FastAPI Replicas (GPU-accelerated)
  - Resources: 6 CPU, 12Gi RAM, 1 GPU per worker
  - 2 worker replicas

**Storage:**
- Persistent Volume Claim: `ray-storage-pvc` (500Gi)
- Mounted at `/mnt/ray-storage` for model caching

**Container Registry:**
- Local registry: `192.168.1.3:5000`
- Image: `embedding-service:v4.0.2`

> **Note:** Architecture diagrams with Mermaid syntax are available in the repository README.md but may not render in Confluence. View them in the GitHub repository or render locally.

### Component Architecture

**FastAPI Application Layer:**
- Entry point: `create_app()` in `src/service/main.py`
- Lifespan management for startup/shutdown
- Router includes: embeddings, similarity, models, health

**Middleware Stack:**
- CORSMiddleware
- GZipMiddleware
- LoggingMiddleware
- ErrorHandlerMiddleware
- RequestValidationMiddleware

**Core Components:**

**EmbeddingEngineManager** (Singleton):
- Manages EmbeddingEngine instances
- Lazy-loads models on demand
- Thread-safe model caching
- Methods: `get_instance()`, `load_model()`, `get_engine()`, `cleanup()`

**EmbeddingEngine:**
- Core embedding computation
- PyTorch model with CUDA acceleration
- HuggingFace Transformers integration
- Thread-safe tokenization
- Key methods:
  - `embed()` - Generate embeddings
  - `embed_with_segments()` - Handle long texts
  - `similarity()` - Compute cosine similarity
  - `_segment_text()` - Split long documents
  - `_process_batch()` - GPU batch processing
  - `_max_pooling_across_segments()` - Aggregate segments

**Dependencies:**
- PyTorch for GPU acceleration and model inference
- HuggingFace Transformers for pre-trained models
- NumPy for array operations and similarity computation

### Request Flow

**Client Request:**
1. Client sends POST request to `/embeddings`
2. NGINX Ingress validates SSL/TLS and API Key
3. Request forwarded to FastAPI application

**Processing:**
4. FastAPI acquires semaphore (concurrency control)
5. `EmbeddingEngineManager.get_engine()` retrieves or loads model
6. If model not loaded: Initialize model on GPU
7. If model cached: Return cached engine instance

**Embedding Generation:**
8. `EmbeddingEngine.embed()` called with texts
9. `_segment_text()` splits long texts into chunks
10. `_tokenize()` performs thread-safe tokenization
11. `_process_batch()` executes GPU forward pass:
    - Transformer model inference
    - Mean pooling across tokens
    - L2 normalization
12. `_max_pooling_across_segments()` aggregates segment embeddings

**Response:**
13. Semaphore released
14. JSON response with embeddings (GZip compressed if >1KB)
15. NGINX forwards HTTPS response to client

### Text Segmentation Flow

For documents exceeding 512 tokens:

**Step 1: Input Text**
- Example: 2048 tokens

**Step 2: Segmentation (`_segment_text()`)**
- chunk_size: 512 tokens
- overlap: 50 tokens
- Creates segments: [0:512], [462:974], [924:1436], [1386:1898], [1848:2048]

**Step 3: Batch Processing (`_process_flat_segments()`)**
- All segments processed on GPU in batches
- batch_size: 1024

**Step 4: Segment Embeddings**
- Each segment produces 384-dimensional vector
- Matrix shape: [5, 384] for 5 segments

**Step 5: Aggregation (`_max_pooling_across_segments()`)**
- Take maximum value per dimension across all segments
- Preserves strongest semantic signals

**Step 6: Final Embedding**
- Single 384-dimensional vector representing entire document

---

## How It Works

EmbeddingService transforms natural language text into dense numerical vectors (embeddings) that capture semantic meaning. These embeddings enable AI applications to understand similarity between texts, perform semantic search, cluster documents, and power retrieval-augmented generation (RAG) systems.

### The Embedding Pipeline

When you send text to the service, it flows through several sophisticated layers:

#### 1. Request Handling

Location: `src/service/routes/embeddings.py`

The FastAPI application receives your HTTP request at the `/embeddings` endpoint. The request is validated by `RequestValidationMiddleware` to ensure it doesn't exceed configured limits (8,192 characters per text, 1,024 texts per batch). The middleware stack also handles CORS, logging, error handling, and GZip compression for large responses.

#### 2. Concurrency Control

Location: `src/service/core/dependencies.py`

Before processing begins, the request must acquire a semaphore from the global semaphore pool. This prevents the service from being overwhelmed by concurrent requests that could exhaust GPU memory. The `get_embedding_semaphore()` function provides access to a semaphore with a limit of 64 concurrent embedding operations (configurable in `src/config.py`). This ensures stable throughput even under burst traffic.

#### 3. Model Management

Location: `src/service/core/engine_manager.py`

The `EmbeddingEngineManager` singleton manages the lifecycle of embedding models. When you request embeddings with a specific model (like "all-mini"), the manager checks if that model is already loaded in its internal cache (`_engines` dictionary). If not, it instantiates a new `EmbeddingEngine` and caches it for future requests. This lazy-loading strategy minimizes memory usage while maintaining fast response times for frequently-used models.

#### 4. Text Segmentation

Location: `src/components/embedding_engine.py` - `_segment_text()`

If your text exceeds 512 tokens (the maximum sequence length for most BERT models), the `EmbeddingEngine` automatically segments it into overlapping chunks. The `_segment_text()` method tokenizes your text, then splits it into segments with 50-token overlaps (configurable). This overlap ensures that semantic information at chunk boundaries isn't lost. For example, a 2,048-token document becomes 5 segments: [0:512], [462:974], [924:1436], [1386:1898], [1848:2048].

#### 5. GPU-Accelerated Processing

Location: `embedding_engine.py` - `_process_batch()` and `_process_flat_segments()`

The segmented texts (or original texts if segmentation isn't needed) are batched and sent to the GPU for processing. The `_process_batch()` method:

- Tokenizes texts using the HuggingFace `AutoTokenizer`
- Pads sequences to equal length for efficient batch processing
- Transfers tensors to the GPU (`torch.device('cuda')`)
- Runs a forward pass through the transformer model (`AutoModel`)
- Applies mean pooling across token embeddings to get sentence-level vectors
- Optionally normalizes embeddings for cosine similarity

For segmented texts, `_process_flat_segments()` batches all segments across all input texts together, maximizing GPU utilization. This is more efficient than processing each text's segments separately.

#### 6. Segment Aggregation

Location: `embedding_engine.py` - `_max_pooling_across_segments()`

After generating embeddings for each segment, the engine aggregates them into a single embedding per text using max pooling. For each dimension in the embedding vector, it takes the maximum value across all segments. This preserves the strongest semantic signals from any part of the document while maintaining a fixed-size output regardless of input length.

#### 7. Response Formation

Location: `src/service/routes/embeddings.py`

The NumPy array of embeddings is converted to a JSON-serializable format and wrapped in a response schema (`EmbeddingResponse` from `src/service/schemas/responses.py`). The response includes the embeddings, their shape, the model name, and processing time. If the response is large, the `GZipMiddleware` compresses it before sending.

### Model Loading and Caching

The `EmbeddingEngine` class (`src/components/embedding_engine.py`) uses HuggingFace Transformers to load pre-trained models. When initialized:

1. It resolves the model name using the `SMALL_MODELS` dictionary (e.g., "all-mini" → "sentence-transformers/all-MiniLM-L6-v2")
2. Downloads the model from HuggingFace Hub if not cached locally
3. Moves the model to the GPU and sets it to evaluation mode
4. Initializes a thread pool executor for async operations
5. Determines the embedding dimension by running a test input

Models are cached in `/root/.cache/huggingface` (controlled by the `TRANSFORMERS_CACHE` environment variable) and persist across container restarts if mounted to a persistent volume.

### Async Architecture

The service uses Python's `asyncio` for non-blocking request handling. While embedding computation itself is synchronous (PyTorch operations), the `EmbeddingEngine.embed()` method wraps synchronous calls with `loop.run_in_executor()`, allowing the event loop to handle other requests while embeddings are computed in a thread pool. This architecture, combined with semaphore-based concurrency control, enables the service to maintain high throughput without blocking.

### Ray Serve Integration

In production (`src/service/ray_deployment.py`), the FastAPI application is deployed using Ray Serve, a scalable serving framework. Ray Serve:

- Creates multiple replicas of the application across GPU workers
- Load-balances requests across replicas
- Provides health checks and auto-recovery
- Scales replicas based on resource availability
- Handles graceful shutdown and model cleanup

The RayService Kubernetes resource defines a Ray cluster with a head node (for coordination) and GPU worker nodes (for inference). Each worker runs replicas of the embedding service, allowing horizontal scaling.

---

## Features

### Core Features

- **GPU Acceleration**: Leverages NVIDIA T4/A4500 GPUs for high-throughput inference via PyTorch CUDA backend
- **Automatic Text Segmentation**: Handles documents up to 8,192 characters with intelligent chunking and overlap (`_segment_text()`)
- **Batch Processing**: Process up to 1,024 texts in a single request with GPU-optimized batching (`_process_flat_segments()`)
- **Multiple Models**: Support for MiniLM, BERT variants, Legal-BERT via HuggingFace Transformers (`SMALL_MODELS` dict)
- **Normalized Embeddings**: L2-normalized vectors for cosine similarity (`normalize_embeddings=True`)
- **Async Processing**: Non-blocking async/await with semaphore-based concurrency control (`get_embedding_semaphore()`)

### Enterprise Features

- **SSL/TLS Encryption**: Let's Encrypt certificates with automatic renewal via cert-manager
- **API Key Authentication**: Secure API access with X-API-Key header validation in middleware
- **Rate Limiting**: 600 requests/minute per IP (configurable in `service.rate_limit_per_minute`)
- **Health Checks**: Liveness (`/health/live`) and readiness (`/health/ready`) probes for Kubernetes
- **Monitoring**: Performance metrics endpoint (`/metrics`), request logging via `LoggingMiddleware`, GPU utilization tracking
- **Error Handling**: Structured error responses with retry logic (`ErrorHandlerMiddleware` and custom exceptions in `core/exceptions.py`)
- **GZip Compression**: Automatic response compression for large payloads via `GZipMiddleware` (>1KB threshold)

### Scalability

- **Ray Serve**: Horizontal scaling with multiple replicas (`ray_deployment.py` deploys FastAPI app across workers)
- **Auto-scaling**: Kubernetes HPA based on CPU/memory/GPU metrics
- **Resource Management**: Semaphore-based concurrency limits prevent GPU OOM (`embedding_concurrency_limit`, `gpu_ops_concurrency_limit`)
- **Graceful Shutdown**: Clean termination with 120s grace period (lifecycle hooks in `rayservice-local.yaml`)
- **PVC Support**: Persistent model caching across restarts (`TRANSFORMERS_CACHE` mounted to persistent volume)

### Advanced Processing Features

**Intelligent Text Segmentation**: The `_segment_text()` method in `EmbeddingEngine` handles long documents by splitting them into overlapping chunks. This is crucial for legal documents, research papers, or any text exceeding the model's 512-token limit. The overlap (default 50 tokens) ensures contextual information at boundaries is preserved.

**Max Pooling Aggregation**: When a text is segmented, each segment gets its own embedding. The `_max_pooling_across_segments()` method aggregates these into a single embedding by taking the maximum value for each dimension across all segments. This captures the strongest semantic signals from the entire document while maintaining a consistent output shape.

**Dynamic Batch Sizing with GPU Memory Management**: The `_process_flat_segments()` method includes automatic retry logic with exponentially decreasing batch sizes. If a CUDA out-of-memory error occurs, it halves the batch size and retries (up to 6 attempts). This ensures robustness even when processing unexpectedly large batches or complex texts that consume more GPU memory.

**Thread-Safe Tokenization**: HuggingFace tokenizers are not thread-safe by default. The `EmbeddingEngine` uses a `_tokenizer_lock` (threading.Lock) to guard all tokenizer operations, preventing race conditions when multiple requests are processed concurrently in the thread pool executor.

**Model Fallback**: If the requested model fails to load (e.g., network issues, invalid model name), the `_load_model()` method automatically falls back to a reliable default model (`distilbert-base-uncased` by default, configurable via `embedding.fallback_model`). This ensures the service remains available even if a specific model is unavailable.

---

## Deployment Environment

### Local Infrastructure

| Resource | Type | Specifications |
|----------|------|----------------|
| **Kubernetes Cluster** | Single-node | v1.32.9 (server node) |
| **System Resources** | Control plane | As needed for cluster management |
| **GPU Resources** | Worker nodes | 2x NVIDIA A4500 (21GB VRAM each) |
| **Container Registry** | Local registry | 192.168.1.3:5000 |
| **Storage** | Local storage | Provisioned as needed |

### Kubernetes Deployment

**Current Version**: v4.0.2
**Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Image**: `192.168.1.3:5000/embedding-service:v4.0.2`

**Pods:**

**Head Pod:**
- Resources: 4 CPU, 8Gi memory
- Node: system pool
- Ports: 6379 (GCS), 8265 (Dashboard), 8000 (API)

**Worker Pods (2 replicas):**
- Resources: 6 CPU, 12Gi memory, 1 GPU
- Node: GPU pool
- GPU: NVIDIA A4500 (21GB VRAM)
- Tolerations: `nvidia.com/gpu=present`

**Services:**
- `embedding-service-serve-svc` (ClusterIP on port 8000)

**Ingresses:**
- `embedding-service-ingress`: http://embeddings.192-168-1-3.nip.io
- `ray-dashboard-ingress`: http://dashboard.192-168-1-3.nip.io

**SSL/TLS:**
- Managed via NGINX Ingress Controller
- Configured with hostNetwork for direct port 80 binding

**Storage:**
- PVC: `ray-storage-pvc` (500Gi)
- Mount path: `/mnt/ray-storage`
- Purpose: Model caching and persistence

---

## API Endpoints

All endpoints are defined in the `src/service/routes/` directory, with request/response schemas in `src/service/schemas/`. The routing is registered in `src/service/main.py` via FastAPI's `include_router()`.

### Embeddings

**POST /embeddings**

Location: `src/service/routes/embeddings.py`

Generate embeddings for single or multiple texts. The endpoint validates input via `EmbeddingRequest` schema, acquires a semaphore to control concurrency, retrieves or loads the requested model via `EmbeddingEngineManager.get_engine()`, calls `engine.embed()` for processing, and returns results wrapped in `EmbeddingResponse` schema.

```bash
curl -X POST http://embeddings.192-168-1-3.nip.io/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Semantic search"],
    "model_name": "all-mini",
    "use_segmentation": true
  }'
```

**Response:**
```json
{
  "embeddings": [[0.12, -0.45, ...], [0.18, -0.32, ...]],
  "shape": [2, 384],
  "model": "all-mini",
  "processing_time": 0.006
}
```

### Similarity

**POST /similarity**

Location: `src/service/routes/similarity.py`

Compute cosine similarity between text pairs. This endpoint generates embeddings for two sets of texts and computes their dot product (since embeddings are L2-normalized, dot product equals cosine similarity). The `similarity()` method in `EmbeddingEngine` handles this efficiently by normalizing embeddings if needed and using NumPy's `np.dot()` for the similarity matrix.

```bash
curl -X POST http://embeddings.192-168-1-3.nip.io/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts1": ["cat on mat"],
    "texts2": ["feline on rug", "dog in park"]
  }'
```

**Response:**
```json
{
  "similarities": [[0.87, 0.23]],
  "shape": [1, 2],
  "processing_time": 0.012
}
```

### Health Checks

All health endpoints are implemented in `src/service/routes/health.py`.

**GET /health**

Service health status with detailed information including GPU availability, loaded models, and system resources.

```bash
curl http://embeddings.192-168-1-3.nip.io/health
```

**GET /health/ready**

Readiness probe for Kubernetes. Returns 200 if the service is ready to accept traffic (checks that at least one model can be loaded successfully). Used by K8s readiness probes to determine when a pod should receive traffic.

**GET /health/live**

Liveness probe for Kubernetes. Returns 200 if the service process is running. Used by K8s to determine when to restart a pod.

### Models

Endpoints for model management are in `src/service/routes/models.py`.

**GET /models**

List available models from the `SMALL_MODELS` dictionary in `EmbeddingEngine`, showing which models can be requested and their HuggingFace paths.

**POST /models/load**

Pre-load a specific model into memory. This is useful for warming up the service before receiving production traffic. The endpoint calls `EmbeddingEngineManager.load_model()` which instantiates an `EmbeddingEngine`, downloads the model if needed, and caches it in the manager's `_engines` dictionary.

**GET /models/info**

Get detailed information about a loaded model including device (CPU/GPU), embedding dimensions, batch size, and GPU memory statistics. This calls `engine.get_model_info()` which introspects the PyTorch model and device configuration.

---

## Installation & Setup

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd EmbeddingService

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run service locally
python -m src.service.main

# Or with uvicorn
uvicorn src.service.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Build

```bash
# Build image
echo "v4.0.2" > version
docker build -t embedding-service:v4.0.2 .

# Run locally
docker run -p 8000:8000 --gpus all embedding-service:v4.0.2

# Tag and push to local registry
docker tag embedding-service:v4.0.2 192.168.1.3:5000/embedding-service:v4.0.2
docker push 192.168.1.3:5000/embedding-service:v4.0.2
```

### Kubernetes Deployment

See the [QUICKSTART.md](QUICKSTART.md) for detailed deployment instructions.

**Quick deploy:**

```bash
# One-command deployment (installs KubeRay, ingress, storage, service)
make quickstart

# Or manual steps:
kubectl apply -f k8s/rayservice-local.yaml
kubectl apply -f k8s/ingress/embedding-service-ingress.yaml
kubectl apply -f k8s/ingress/ray-dashboard-ingress.yaml

# Check deployment status
kubectl get pods
kubectl get rayservice embedding-service
```

---

## Configuration

### Model Configuration

Edit `src/config.py`:

```python
defaults = {
    "embedding": {
        "default_model": "all-mini",  # MiniLM (384 dim)
        "batch_size": 1024,
        "max_length": 512,
        "normalize_embeddings": True,
        "device": "cuda",
        "use_segmentation": True,
        "segment_overlap": 50,
        "fallback_model": "distilbert-base-uncased"
    },
    "service": {
        "max_text_length": 8192,
        "max_batch_size": 1024,
        "embedding_concurrency_limit": 64,
        "similarity_concurrency_limit": 16
    }
}
```

### Available Models

| Key | Model Path | Dimensions | Use Case |
|-----|-----------|------------|----------|
| `all-mini` | sentence-transformers/all-MiniLM-L6-v2 | 384 | General purpose (recommended) |
| `legal-bert` | nlpaueb/legal-bert-base-uncased | 768 | Legal documents |
| `distilbert` | distilbert-base-uncased | 768 | Fallback model |
| `bert-small` | google/bert_uncased_L-4_H-512_A-8 | 512 | Medium accuracy/speed |

### Environment Variables

```bash
# Model caching
TRANSFORMERS_CACHE=/root/.cache/huggingface

# Ray storage
RAY_STORAGE_PATH=/mnt/ray-storage

# GPU visibility
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
```

---

## Development

### Project Structure

```
EmbeddingService/
├── src/
│   ├── components/
│   │   └── embedding_engine.py      # Core embedding engine
│   ├── service/
│   │   ├── main.py                  # FastAPI application
│   │   ├── ray_deployment.py        # Ray Serve deployment
│   │   ├── core/
│   │   │   ├── engine_manager.py    # Singleton manager
│   │   │   ├── dependencies.py      # DI & semaphores
│   │   │   └── exceptions.py        # Custom exceptions
│   │   ├── routes/
│   │   │   ├── embeddings.py        # Embedding endpoints
│   │   │   ├── similarity.py        # Similarity endpoints
│   │   │   ├── models.py            # Model management
│   │   │   └── health.py            # Health checks
│   │   ├── schemas/
│   │   │   ├── requests.py          # Request models
│   │   │   └── responses.py         # Response models
│   │   └── middleware/
│   │       ├── error_handler.py     # Error handling
│   │       ├── logging.py           # Request logging
│   │       └── validation.py        # Input validation
│   ├── config.py                    # Configuration
│   └── config_handler.py            # Config helper
├── k8s/
│   ├── rayservice-local.yaml        # Local K8s deployment
│   ├── rayservice-aks-*.yaml        # AKS deployments
│   └── ingress/
│       ├── embedding-service-ingress.yaml  # API ingress
│       └── ray-dashboard-ingress.yaml      # Dashboard ingress
├── helm/
│   ├── ray-plain-cluster/           # Plain Ray cluster chart
│   └── ray-serving-cluster/         # Ray Serve cluster chart
├── tests/                           # Unit tests
├── Dockerfile                       # Main image
├── requirements.txt                 # Dependencies
├── Makefile                         # Deployment automation
├── QUICKSTART.md                    # Quick deployment guide
└── setup.py                         # Package setup
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_embedding_engine.py

# Run with coverage
pytest --cov=src tests/
```

### Making Changes

1. **Change Model**: Update `src/config.py` → `defaults["embedding"]["default_model"]` to one of the keys in `EmbeddingEngine.SMALL_MODELS`
2. **Version Bump**: Update `version` file (e.g., `echo "v4.0.3" > version`)
3. **Rebuild**: `make build-push` or manually:
   ```bash
   docker build --no-cache -t 192.168.1.3:5000/embedding-service:v4.0.3 .
   docker push 192.168.1.3:5000/embedding-service:v4.0.3
   ```
4. **Update K8s**: Edit `k8s/rayservice-local.yaml` to update both head and worker container image tags
5. **Deploy**: `make redeploy` or manually `kubectl apply -f k8s/rayservice-local.yaml`
6. **Verify**: Check pods with `kubectl get pods` and test API to confirm model change

**Important**: When changing models, use `--no-cache` to avoid Docker layer caching of old model downloads. HuggingFace downloads models at runtime in `_load_model()`, but Docker may cache the layers where those downloads occurred, causing the old model to persist even with new configuration.

### Understanding the Configuration System

The configuration is managed by a custom `Configuration` class in `src/config_handler.py` and instantiated in `src/config.py`. The `CustConf` class extends `Configuration` and provides:

- **Nested Key Access**: Use dot notation like `app_conf.get("embedding.batch_size")` to access nested config values
- **Logging Configuration**: The `init_logger()` method sets up loggers with formats and handlers defined in config
- **Type Safety**: Configuration values have defaults, so `get()` always returns a valid value
- **Module-Level Access**: Import via `from src.config import app_conf` throughout the codebase

The `Base` class in `config_handler.py` is a mixin that provides `_app_config` and `_logger` to any class that inherits it. This is why `EmbeddingEngine(Base)` has access to configuration without explicitly passing it.

### Middleware Pipeline

Middleware executes in reverse order of registration (last added is first executed). The pipeline in `main.py` is:

1. **ErrorHandlerMiddleware** (outermost) - Catches all exceptions and converts to JSON responses
2. **LoggingMiddleware** - Logs request/response details with timing
3. **GZipMiddleware** - Compresses large responses (>1KB)
4. **RequestValidationMiddleware** - Validates request size and content
5. **CORSMiddleware** - Handles cross-origin requests (innermost, closest to routes)

This ordering ensures errors are caught and formatted properly, logging captures the full request lifecycle, and validation happens before processing.

---

## Monitoring & Health

### Health Endpoints

- **Liveness**: `/health/live` - Pod is running
- **Readiness**: `/health/ready` - Pod can serve traffic (model loaded)
- **Detailed Health**: `/health` - Full status with GPU info

### Metrics

Access `/metrics` for:
- Request counts and error rates
- Average response times
- Per-endpoint statistics
- Semaphore usage

### Ray Dashboard

Access at http://dashboard.192-168-1-3.nip.io:
- Cluster status
- Resource utilization
- Serve deployment metrics
- Logs and traces

### Kubernetes Monitoring

```bash
# Pod status
kubectl get pods

# Logs
kubectl logs <pod-name>

# Resource usage
kubectl top pods

# Describe pod
kubectl describe pod <pod-name>

# Ray cluster status
kubectl get rayservice embedding-service

# Check deployment status
make k8s-status
```

---

## Performance

### Benchmarks (v4.0.2 with MiniLM on A4500 GPU)

| Metric | Value |
|--------|-------|
| **Embedding Dimensions** | 384 |
| **Processing Time** | ~6ms per text |
| **Throughput** | 7-8 requests/sec (batch processing) |
| **Batch Size** | Up to 1,024 texts |
| **GPU Utilization** | ~80% during burst |
| **Model Load Time** | ~2-3 seconds (first request) |

### Optimization Tips

1. **Batch Requests**: Send multiple texts in one request (up to 1,024) for better GPU utilization
2. **Use Segmentation**: Enable for long documents (>512 tokens) to avoid truncation
3. **Model Selection**: MiniLM (384 dim) is 73% faster than Legal-BERT (768 dim)
4. **GPU Warmup**: First request takes longer due to model loading - pre-warm with `/models/load`
5. **Connection Pooling**: Reuse HTTP connections for better throughput

### Scaling

To handle higher load:

```bash
# Scale GPU workers (edit rayservice-local.yaml)
kubectl patch rayservice embedding-service --type=json \
  -p='[{"op": "replace", "path": "/spec/rayClusterConfig/workerGroupSpecs/0/replicas", "value": 3}]'

# Or use Makefile
make redeploy

# Check new replicas
kubectl get pods -l ray.io/is-ray-node=yes
```

---

## Makefile Commands

The project includes a comprehensive Makefile for common operations:

### Python Development
- `make pyenv` - Create Python virtual environment
- `make install` - Install dependencies
- `make test` - Run tests
- `make check` - Run pre-commit checks

### Docker
- `make docker-build` - Build Docker image
- `make docker-push` - Push to registry
- `make build-push` - Build and push

### Kubernetes Deployment
- `make quickstart` - Complete setup (KubeRay + Ingress + Service + Test)
- `make k8s-install-kuberay` - Install KubeRay operator
- `make k8s-install-ingress` - Install NGINX Ingress
- `make k8s-deploy-rayservice` - Deploy RayService
- `make k8s-apply-ingress` - Apply ingress manifests
- `make k8s-status` - Show deployment status
- `make k8s-test` - Test API via ingress
- `make k8s-delete` - Delete RayService
- `make redeploy` - Delete and redeploy

See [Makefile](Makefile) for full list of targets and [QUICKSTART.md](QUICKSTART.md) for deployment guide.

---

## Troubleshooting

### Common Issues

**Pods not scheduling:**
```bash
# Check node taints
kubectl describe nodes | grep -i taint

# Remove disk-pressure taint if present
kubectl taint nodes <node-name> node.kubernetes.io/disk-pressure:NoSchedule-

# Check resources
kubectl describe nodes | grep -A 5 "Allocated resources"
```

**Service not responding:**
```bash
# Check RayService status
kubectl describe rayservice embedding-service

# Check operator logs
kubectl logs -n kuberay-system -l app.kubernetes.io/name=kuberay-operator

# Check head pod logs
kubectl logs $(kubectl get pod -l ray.io/node-type=head -o name)
```

**Ingress not accessible:**
```bash
# Verify hostNetwork enabled
kubectl get deployment ingress-nginx-controller -n ingress-nginx \
  -o jsonpath='{.spec.template.spec.hostNetwork}'
# Should output: true

# Test direct pod access (bypass ingress)
POD_IP=$(kubectl get endpoints embedding-service-serve-svc -o jsonpath='{.subsets[0].addresses[0].ip}')
curl http://$POD_IP:8000/health
```

**GPU not available:**
```bash
# Check GPU resources
nvidia-smi

# Check pod GPU allocation
kubectl describe pod <worker-pod-name> | grep -A 5 Limits
```

---

## Support

For issues, questions, or contributions:
- **Repository**: https://github.com/your-org/EmbeddingService
- **Documentation**: See QUICKSTART.md and code comments
- **Issues**: Open GitHub issues for bugs or feature requests
