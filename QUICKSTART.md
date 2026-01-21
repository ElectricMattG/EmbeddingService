# EmbeddingService Quick Start

This guide walks you through deploying the EmbeddingService—a FastAPI-based text embedding service wrapped with Ray Serve for distributed, autoscaling inference. The service uses sentence-transformers models and is designed to run on Kubernetes using the KubeRay operator.

## What This Service Does

EmbeddingService provides REST API endpoints for:
- **Text embeddings**: Convert text into dense vector representations using transformer models
- **Similarity computation**: Calculate semantic similarity between text pairs
- **Multi-model support**: Switch between different embedding models dynamically
- **Batch processing**: Handle large batches efficiently with GPU acceleration

The service is built with:
- **FastAPI** for the HTTP API layer
- **Ray Serve** for distributed serving and autoscaling (2-16 replicas)
- **PyTorch + Transformers** for model inference
- **KubeRay** for Kubernetes-native Ray cluster management

## Prerequisites

- Kubernetes cluster running (tested on v1.32+)
- `kubectl` configured and `helm` installed
- Docker image built and available (see Building section)
- GPU nodes with NVIDIA drivers (optional but recommended for performance)

## Quick Deploy with Makefile

The fastest way to deploy everything from scratch. This single command handles the complete setup process, installing all dependencies and configuring ingress access automatically.

```bash
# One command to install KubeRay, ingress, storage, RayService, and test
make quickstart
```

**What this does:**
1. **Installs KubeRay operator** (v1.4.2) - Manages Ray cluster lifecycle on Kubernetes
2. **Installs NGINX Ingress Controller** - Routes external traffic to the service, configured with `hostNetwork=true` for single-node clusters
3. **Creates storage PVC** - 500Gi persistent volume for caching downloaded models (avoids re-downloading on pod restarts)
4. **Deploys RayService** - Applies the main deployment manifest with 1 head node + 2 GPU workers
5. **Applies ingress manifests** - Sets up DNS-based routing using nip.io for easy local access
6. **Runs health check** - Verifies the service is responding correctly

This approach is ideal for first-time setup or clean deployments.

## Manual Deployment Steps

For more control or when deploying to existing infrastructure, follow these individual steps. Each step can be run independently if certain components are already installed.

### 1. Install KubeRay Operator

The KubeRay operator is a Kubernetes controller that manages Ray cluster resources (RayCluster, RayService, RayJob). It watches for these custom resources and orchestrates the creation/deletion of Ray head and worker pods.

```bash
make k8s-install-kuberay

# Or manually:
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.4.2&timeout=90s"
kubectl wait --for=condition=available --timeout=300s deployment/kuberay-operator -n kuberay-system
```

**Why KubeRay?** It handles the complexity of Ray cluster management: health checks, autoscaling, rolling updates, and integration with Kubernetes primitives.

### 2. Install NGINX Ingress Controller

The ingress controller routes external HTTP(S) traffic to services inside the cluster. We configure it with `hostNetwork=true` to bind directly to port 80 on the host, avoiding NodePort complexity on single-node setups.

```bash
make k8s-install-ingress

# Or manually with Helm:
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace \
  --set controller.hostNetwork=true \
  --set controller.kind=DaemonSet
```

**DaemonSet mode** ensures one ingress controller pod runs on each node, ideal for bare-metal or edge deployments.

### 3. Create Storage PVC

Models downloaded by Hugging Face Transformers are cached to avoid re-downloading on every pod restart. This PVC is mounted at `/mnt/ray-storage` on all Ray pods.

```bash
make k8s-apply-storage

# This creates a 500Gi PVC named 'ray-storage-pvc' for model caching
```

**Note:** Adjust the storage size based on the number of models you plan to use. Each model can be 100MB-5GB.

### 4. Deploy RayService

This is the core deployment step. The RayService CRD defines both the Ray cluster configuration (head/workers) and the Ray Serve application (your FastAPI service).

```bash
make k8s-deploy-rayservice

# Or manually:
kubectl apply -f k8s/rayservice-local.yaml
kubectl wait --for=condition=ready pod -l ray.io/is-ray-node=yes --timeout=600s
```

**RayService manifest** (`k8s/rayservice-local.yaml`) contains:
- **Head node spec**: 4 CPU, 8Gi RAM (handles coordination, no GPU needed)
- **Worker group spec**: 2 replicas, 6 CPU, 12Gi RAM, 1 GPU each (handles model inference)
- **Serve config**: Points to `src.service.ray_deployment:deployment` (the FastAPI app wrapper)
- **Autoscaling config**: 2-16 replicas, 0.2 GPU per replica, scales based on request queue depth

Expected output:
```
NAME                                 READY   STATUS    RESTARTS   AGE
embedding-service-xxxxx-head-xxxxx   1/1     Running   0          2m
embedding-service-xxxxx-worker-xxx   1/1     Running   0          2m
embedding-service-xxxxx-worker-xxx   1/1     Running   0          2m
```

### 5. Apply Ingress Manifests

Creates two ingress resources: one for the API service and one for the Ray dashboard. Both use nip.io DNS for automatic local resolution (e.g., `192-168-1-3.nip.io` resolves to `192.168.1.3`).

```bash
make k8s-apply-ingress

# Or manually:
kubectl apply -f k8s/ingress/embedding-service-ingress.yaml
kubectl apply -f k8s/ingress/ray-dashboard-ingress.yaml
```

**Ingress features configured:**
- 20MB max body size for large batch requests
- 180s timeout for long-running inference jobs
- Proxy buffering disabled for streaming responses

### 6. Verify Deployment

Test that the service is accessible and responding correctly.

```bash
make k8s-test

# Or manually test API endpoint:
curl http://embeddings.192-168-1-3.nip.io/health | jq .

# Test embeddings generation:
curl -X POST http://embeddings.192-168-1-3.nip.io/embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "model": "sentence-transformers/all-MiniLM-L6-v2"}'
```

**Expected response:** A 384-dimensional embedding vector for the default model, or 768-dimensional for larger models like BERT.

## Service Endpoints

- **API Service**: `http://embeddings.192-168-1-3.nip.io/` (replace with your NODE_IP)
- **Ray Dashboard**: `http://dashboard.192-168-1-3.nip.io/` (replace with your NODE_IP)
- **API Docs**: `http://embeddings.192-168-1-3.nip.io/docs`
- **Health Check**: `http://embeddings.192-168-1-3.nip.io/health`

> **Note**: Replace `192-168-1-3` with your actual node IP with dashes (e.g., for `10.0.0.5` use `10-0-0-5`)

## Check Deployment Status

```bash
make k8s-status

# Shows:
# - RayService status
# - Ray pods
# - Services
# - Ingress resources
```

## Building and Pushing Docker Image

Before deploying, you need a Docker image with your application code and dependencies.

```bash
# Build image (tags with version from ./version file)
make docker-build

# Push to registry
make docker-push

# Or do both:
make build-push
```

**Image configuration:**
- **Registry**: Configurable via `REGISTRY` env var (default: `192.168.1.3:5000`)
- **Image name**: `embedding-service`
- **Version**: Read from `./version` file (e.g., `v4.0.2`)
- **Base image**: Python 3.12 with CUDA support
- **Size**: ~8GB (includes PyTorch, Transformers, Ray)

**To update the image:**
1. Make code changes in `src/`
2. Optionally update `./version` file
3. Run `make build-push`
4. Run `make redeploy` to update the cluster

The Dockerfile multi-stage build optimizes for:
- Minimal layer count (cached pip installs)
- Non-root user for security
- Health check endpoints
- Graceful shutdown handling

## Repository Structure

Understanding the repository layout helps you navigate and customize the deployment:

```
k8s/                          # Raw Kubernetes manifests (used by quickstart)
  ├── rayservice-local.yaml   # Main RayService deployment for local/dev
  ├── rayservice-aks-*.yaml   # AKS-specific deployments (Azure)
  ├── ingress/                # Ingress resources
  │   ├── embedding-service-ingress.yaml  # API ingress (port 80)
  │   └── ray-dashboard-ingress.yaml      # Dashboard ingress (port 8265)
  └── devops/                 # CI/CD configs (ArgoCD, Jenkins)

helm/                         # Helm charts (alternative deployment method)
  ├── ray-plain-cluster/      # Plain Ray cluster chart (no Serve)
  ├── ray-serving-cluster/    # Ray Serve cluster chart (templated)
  └── values-embedding-service.yaml  # Values for Helm deployments

src/                          # Python application code
  ├── service/                # FastAPI service and Ray integration
  │   ├── main.py            # FastAPI app definition
  │   ├── ray_deployment.py  # Ray Serve wrapper with @serve.deployment
  │   ├── core/              # Engine manager, dependencies
  │   ├── middleware/        # Error handling, logging, validation
  │   ├── routes/            # API route handlers
  │   └── schemas/           # Pydantic models for request/response
  ├── components/            # Core embedding engine
  └── config.py              # Configuration management

tests/                        # Test scripts and load testing tools
docs/                         # Deployment guides and troubleshooting
```

**Key files to customize:**
- `k8s/rayservice-local.yaml` - Adjust worker count, GPU allocation, resource limits
- `k8s/ingress/*.yaml` - Change hostnames or timeout values
- `src/config.py` - Modify concurrency limits, default models
- `Makefile` - Add custom deployment targets or modify existing ones

## Architecture

This service uses a distributed architecture optimized for GPU-accelerated inference:

### Node Configuration
- **Head Node**: 4 CPU, 8Gi RAM, no GPU
  - Runs Ray GCS (Global Control Service) for cluster coordination
  - Hosts the Ray dashboard on port 8265
  - Lightweight, handles scheduling and metadata only

- **Worker Nodes**: 2 replicas, 6 CPU, 12Gi RAM, 1 GPU each
  - Run the actual model inference workloads
  - Each worker can host multiple Ray Serve replicas
  - GPU sharing via fractional allocation (0.2 GPU per replica)

### Ray Serve Autoscaling
- **Min replicas**: 2 (always-on for low latency)
- **Max replicas**: 16 (scales up under load)
- **GPU per replica**: 0.2 (allows 5 concurrent models per GPU)
- **Target**: 8 ongoing requests per replica (scales when queue depth exceeds this)

### Storage
- **Shared PVC** at `/mnt/ray-storage` (500Gi)
- Caches Hugging Face models downloaded on first use
- Persists across pod restarts, avoiding re-downloads
- Shared between head and all workers via ReadWriteMany or NFS

### Request Flow
```
Client → Ingress (port 80) → Service (port 8000) → Ray Serve →
Ray Actor Pool (autoscales) → GPU Inference → Response
```

**Autoscaling triggers:**
- Queue depth > 8 requests/replica → scale up
- Queue empty for 60s → scale down
- Hardware constraint: Max 10 replicas per GPU (0.2 GPU each)

## Makefile Commands

### Deployment
- `make quickstart` - Complete setup from scratch
- `make k8s-deploy-rayservice` - Deploy RayService only
- `make k8s-apply-ingress` - Apply ingress manifests
- `make redeploy` - Delete and redeploy RayService

### Status & Testing
- `make k8s-status` - Show all resource status
- `make k8s-test` - Test API via ingress

### Cleanup
- `make k8s-delete` - Delete RayService and force-delete pods

## Troubleshooting

### Service not starting

```bash
# Check RayService status
kubectl describe rayservice embedding-service

# Check operator logs
kubectl logs -n kuberay-system -l app.kubernetes.io/name=kuberay-operator

# Check head pod logs
kubectl logs $(kubectl get pod -l ray.io/node-type=head -o name)
```

### Pods not scheduling

```bash
# Check node taints
kubectl describe nodes | grep -i taint

# Remove disk-pressure taint if present
kubectl taint nodes <node-name> node.kubernetes.io/disk-pressure:NoSchedule-

# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check disk space
df -h
```

### Ingress not accessible

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx

# Verify hostNetwork is enabled (required for single-node)
kubectl get deployment ingress-nginx-controller -n ingress-nginx \
  -o jsonpath='{.spec.template.spec.hostNetwork}'
# Should output: true

# Test direct pod access (bypass ingress)
POD_IP=$(kubectl get endpoints embedding-service-serve-svc -o jsonpath='{.subsets[0].addresses[0].ip}')
curl http://$POD_IP:8000/health
```

### Image pull errors

```bash
# Verify image exists in registry
curl http://192.168.1.3:5000/v2/embedding-service/tags/list

# Check image pull policy in manifest
grep imagePullPolicy k8s/rayservice-local.yaml
```

## Updating the Service

```bash
# Option 1: Update image version
# Edit version file, rebuild and push
echo "v4.0.3" > version
make build-push
make redeploy

# Option 2: Direct manifest update
# Edit k8s/rayservice-local.yaml, then:
make redeploy
```

## Cleanup

```bash
# Delete RayService
make k8s-delete

# Or manually:
kubectl delete -f k8s/rayservice-local.yaml
kubectl delete -f k8s/ingress/embedding-service-ingress.yaml
kubectl delete -f k8s/ingress/ray-dashboard-ingress.yaml

# Optional: Remove KubeRay operator
kubectl delete -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.4.2&timeout=90s"
```

## Advanced: Using Helm Charts

The `helm/` directory contains Helm charts for more complex or multi-environment deployments. These provide templating and parameterization beyond what raw manifests offer.

```bash
# Deploy using Helm (alternative to raw manifests)
helm install my-ray-cluster helm/ray-serving-cluster \
  --values helm/values-embedding-service.yaml

# Or use ray-plain-cluster for non-serving workloads
helm install my-ray helm/ray-plain-cluster
```

**When to use Helm vs raw manifests:**
- **Raw manifests** (`k8s/`): Simple, direct, easy to debug. Good for dev/test or single deployment.
- **Helm charts** (`helm/`): Templated, reusable across environments (dev/staging/prod). Good for multi-cluster or GitOps workflows.

**Helm chart features:**
- Parameterized worker count, GPU allocation, resource limits
- Environment-specific values files (e.g., `local-aws.yaml` for AWS)
- Optional components (logging PVC, IRSA for AWS, custom ingress templates)

**Note**: The quickstart uses raw K8s manifests from `k8s/`, not Helm charts. Choose the approach that fits your workflow.
