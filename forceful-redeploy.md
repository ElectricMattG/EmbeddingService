# Forceful Redeploy

## Quick Ray Cluster Forced Redeployment Process

### 1. Delete Current Deployment
```bash
kubectl delete raycluster ray-cluster-simple
```

### 2. Force Delete Stuck Pods (if needed)
```bash
# Check for terminating pods
kubectl get pods -l ray.io/cluster=ray-cluster-simple

# Force delete if stuck in Terminating state
kubectl delete pod ray-cluster-simple-head-* ray-cluster-simple-worker-* --force --grace-period=0
```

### 3. Verify Clean State
```bash
kubectl get pods -l ray.io/cluster=ray-cluster-simple
# Should return "No resources found"
```

### 4. Redeploy with Updated Configuration
```bash
kubectl apply -f ray-cluster-dual-workers.yaml
```

### 5. Monitor New Deployment
```bash
# Watch pod creation
kubectl get pods -l ray.io/cluster=ray-cluster-simple -w

# Check specific pod status
kubectl describe pod ray-cluster-simple-head-<pod-id>
```

### 6. Quick Health Check
```bash
# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l ray.io/cluster=ray-cluster-simple --timeout=300s

# Verify all containers running
kubectl get pods -l ray.io/cluster=ray-cluster-simple
```

### Expected Results
- **Head pod**: 2/2 Ready (ray-head + autoscaler containers)
- **Worker pods**: 1/1 Ready (ray-worker containers)  
- **No CrashLoopBackOff** if rayStartParams properly configured

### Troubleshooting
If pods still crash after redeploy:
1. Check autoscaler logs: `kubectl logs <head-pod> -c autoscaler`
2. Verify rayStartParams in YAML configuration
3. Check image pull issues: `kubectl describe pod <pod-name>`