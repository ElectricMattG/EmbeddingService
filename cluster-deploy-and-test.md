# Cluster Deploy and Test

## Ray Cluster Deployment Process

### 1. Pre-Deployment Checklist
- ✅ Docker image built and pushed to registry
- ✅ Kubernetes cluster running with GPU nodes
- ✅ KubeRay operator installed
- ✅ Sufficient disk space (avoid >90% usage)

### 2. Deploy Ray Cluster
```bash
# Apply Ray cluster configuration
kubectl apply -f ray-cluster-dual-workers.yaml

# Monitor deployment
kubectl get pods -l ray.io/cluster=ray-cluster-simple -w
```

### 3. Verify Cluster Health
```bash
# Check all pods are running
kubectl get pods -l ray.io/cluster=ray-cluster-simple

# Expected output:
# ray-cluster-simple-head-xxxxx                1/1     Running
# ray-cluster-simple-worker-gpu-worker-xxxxx   1/1     Running  
# ray-cluster-simple-worker-gpu-worker-xxxxx   1/1     Running

# Check services
kubectl get services -l ray.io/cluster=ray-cluster-simple
```

### 4. Port Forward Dashboard and Endpoints
```bash
# Ray Dashboard (8265)
kubectl port-forward service/ray-cluster-simple-head-svc 8265:8265 &

# Ray Client (10001) 
kubectl port-forward service/ray-cluster-simple-head-svc 10001:10001 &

# Ray Serve (8000) - when application deployed
kubectl port-forward service/ray-cluster-simple-head-svc 8000:8000 &
```

### 5. Test Dashboard Access
```bash
# Test dashboard connectivity
curl -I http://localhost:8265

# Expected: HTTP/1.1 200 OK
# Dashboard URL: http://localhost:8265
```

### 6. Deploy Ray Serve Application
```bash
# Execute deployment script inside cluster
kubectl exec -it ray-cluster-simple-head-xxxxx -c ray-head -- python /workspace/serve_deploy.py

# Or copy and run locally if configured
python serve_deploy.py
```

### 7. Test Ray Serve Endpoint
```bash
# Test embeddings endpoint
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Ray Serve is working"],
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'

# Expected: JSON response with embeddings array
```

### 8. Performance Testing
```bash
# Test large batch processing (max_batch=512)
python test_large_batch.py

# Monitor GPU utilization
kubectl exec -it ray-cluster-simple-worker-gpu-worker-xxxxx -- nvidia-smi

# Check Ray dashboard for metrics
# Navigate to: http://localhost:8265/serve
```

## Troubleshooting

### Cluster Issues
- **CrashLoopBackOff**: Check ISSUE1-autoscaler-params.md
- **Pod stuck in Init**: Check image pull and PVC availability
- **Resource issues**: Review headnode-log-discovery.md

### Port Forward Issues
- **Connection refused on 8000**: Ray Serve not deployed yet
- **Dashboard not accessible**: Check ray-head container health
- **Port conflicts**: Kill existing port-forward processes

### Application Deployment Issues
- **Import errors**: Verify image contains all dependencies
- **GPU not detected**: Check nvidia.com/gpu resource limits
- **Memory issues**: Monitor pod resource usage

## Health Check Commands
```bash
# Quick cluster status
kubectl get raycluster ray-cluster-simple

# Pod health
kubectl get pods -l ray.io/cluster=ray-cluster-simple

# Service endpoints
kubectl get svc -l ray.io/cluster=ray-cluster-simple

# Resource usage
kubectl top pods -l ray.io/cluster=ray-cluster-simple

# Dashboard health
curl -s http://localhost:8265/api/cluster_status | jq .
```

## Expected Results
- **Ray Cluster**: 1 head + 2 GPU workers running
- **Dashboard**: Accessible at http://localhost:8265  
- **Ray Serve**: Embeddings API at http://localhost:8000
- **Performance**: 512 batch size processing with dual GPU support
- **Resource Usage**: ~0.5 GPU per replica, optimized memory allocation

## Cleanup
```bash
# Remove Ray cluster
kubectl delete raycluster ray-cluster-simple

# Stop port forwarding
pkill -f "kubectl port-forward"

# Verify cleanup
kubectl get pods -l ray.io/cluster=ray-cluster-simple
# Should return: No resources found
```