# Headnode Log Discovery

## Quick Ray Cluster Health Check Process

### 1. Access Ray Session Logs
```bash
kubectl exec ray-cluster-simple-head-<pod-id> -c ray-head -- ls /tmp/ray/session_<timestamp>/logs/
```

### 2. Key Log Files & Purpose

| Log File | Purpose | Health Check |
|----------|---------|--------------|
| `gcs_server.out` | Cluster coordination, node registration | Check for registered nodes count |
| `raylet.out` | Worker process management | Look for remote node managers |
| `dashboard.log` | Web UI health | Verify 200 OK health responses |
| `dashboard_agent.log` | Local raylet health | Check raylet health endpoints |
| `monitor.log` | Autoscaler issues | **Critical**: Check for KeyError crashes |
| `runtime_env_agent.out` | Deployment dependencies | Verify running on port 60590 |
| `debug_state_gcs.txt` | Resource allocation summary | Check node/actor/placement counts |

### 3. Critical Commands
```bash
# Check cluster coordination
kubectl exec <head-pod> -c ray-head -- cat /tmp/ray/session_*/logs/gcs_server.out | tail -20

# Check autoscaler (most common failure)
kubectl exec <head-pod> -c ray-head -- tail -20 /tmp/ray/session_*/logs/monitor.log

# Verify worker connectivity
kubectl exec <head-pod> -c ray-head -- cat /tmp/ray/session_*/logs/raylet.out | tail -50

# Check resource state
kubectl exec <head-pod> -c ray-head -- head -50 /tmp/ray/session_*/logs/debug_state_gcs.txt
```

### 4. Common Issues & Fixes

**Autoscaler CrashLoopBackOff:**
- **Symptom**: `KeyError: 'rayStartParams'` in monitor.log
- **Fix**: Add `rayStartParams` to head and worker specs in ray-cluster YAML

**Cluster Health Indicators:**
- ✅ GCS shows 3+ registered nodes
- ✅ Dashboard returns 200 OK every few seconds
- ✅ Raylet detects remote node managers
- ✅ Runtime env agent running on port 60590

### 5. Quick Status Check
```bash
kubectl get pods -l ray.io/cluster=ray-cluster-simple
kubectl get events --sort-by=.metadata.creationTimestamp | tail -5
```

**Healthy cluster**: Ray components running, only autoscaler may need rayStartParams fix.
