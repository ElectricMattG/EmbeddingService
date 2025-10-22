# Embedding Service - Docker Deployment

A GPU-accelerated text embedding service using transformer models, containerized with Docker.

## Quick Start

### Build the Image
```bash
docker build -t embedding-service:latest .
```

### Run with Docker
```bash
docker run -d \
  --name embedding-service \
  --gpus device=0 \
  -p 8090:8090 \
  embedding-service:latest
```

### Run with Docker Compose
```bash
docker-compose up -d
```

## Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)

### Ports
- `8090`: HTTP API endpoint

## API Endpoints

### Health Checks
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /health/status` - Detailed status with GPU info

### Models
- `GET /models/available` - List available models
- `POST /models/load` - Load a specific model

### Embeddings
- `POST /embeddings` - Generate embeddings for text(s)
- `POST /embeddings/quick` - Quick embedding generation

### Similarity
- `POST /similarity` - Compute similarity between texts
- `POST /similarity/search` - Search for most similar texts

## Testing

```bash
# Check health
curl http://localhost:8090/health/live

# Generate embeddings
curl -X POST http://localhost:8090/embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test sentence"]}'
```

## Container Management

```bash
# View logs
docker logs embedding-service

# Follow logs
docker logs -f embedding-service

# Stop container
docker stop embedding-service

# Remove container
docker rm embedding-service

# View stats
docker stats embedding-service
```

## Image Details

- **Base Image**: python:3.12-slim
- **Size**: ~7.9GB (includes PyTorch, Transformers, and dependencies)
- **GPU Support**: NVIDIA CUDA via nvidia-container-toolkit
- **Default Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

## Requirements

- Docker 20.10+
- nvidia-container-toolkit (for GPU support)
- CUDA-compatible GPU

## Development

The container runs in production mode. For development:

```bash
docker run -it \
  --name embedding-service-dev \
  --gpus device=0 \
  -p 8090:8090 \
  -v $(pwd)/src:/app/src \
  embedding-service:latest \
  /bin/bash
```
