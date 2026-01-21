# ISSUE#1: Autoscaler Parameters Issue

## Problem Description
Ray cluster autoscaler container crashes in CrashLoopBackOff when rayStartParams are misconfigured or contain invalid parameters.

## Symptoms
- **Head pod**: 1/2 Ready (ray-head running, autoscaler failing)
- **Worker pods**: Stuck in Init:0/1 (waiting for head node)
- **Autoscaler logs**: `KeyError: 'rayStartParams'` or invalid parameter errors
- **GCS connection**: Autoscaler can't connect to GCS server

## Root Causes

### Cause 1: Missing rayStartParams
```yaml
# Missing rayStartParams section entirely
headGroupSpec:
  serviceType: ClusterIP
  template:
    # ... container spec
```

**Error**: `KeyError: 'rayStartParams'` in autoscaler logs

### Cause 2: Invalid Ray Parameters
```yaml
# Invalid --serve-host parameter
rayStartParams:
  dashboard-host: 0.0.0.0
  serve-host: 0.0.0.0  # ❌ Invalid parameter
  num-cpus: "2"
```

**Error**: `Error: No such option: --serve-host`

### Cause 3: Ray Version Compatibility
Different Ray versions support different parameters. Always check Ray documentation for your specific version.

## Solutions

### Solution 1: Add Valid rayStartParams
```yaml
headGroupSpec:
  rayStartParams:
    dashboard-host: 0.0.0.0
    num-cpus: "2"
    block: "true"
workerGroupSpecs:
  - rayStartParams:
      num-cpus: "3"
      num-gpus: "1"
      block: "true"
```

### Solution 2: Disable Autoscaler (Elimination Strategy)
```yaml
spec:
  enableInTreeAutoscaling: false  # Disable autoscaler completely
  headGroupSpec:
    serviceType: ClusterIP
    # No rayStartParams needed
```

## Debugging Steps
1. **Check autoscaler logs**: `kubectl logs <head-pod> -c autoscaler`
2. **Check ray-head logs**: `kubectl logs <head-pod> -c ray-head`
3. **Verify Ray version**: Check rayVersion in cluster spec
4. **Test parameters**: Use `ray start --help` in container to verify valid options

## Prevention
- Always test rayStartParams with your specific Ray version
- Use minimal required parameters only
- Consider disabling autoscaler if not needed for static workloads
- Validate YAML syntax before deployment

## Quick Fix Commands
```bash
# Remove invalid rayStartParams and disable autoscaler
kubectl delete raycluster ray-cluster-simple
# Edit YAML: set enableInTreeAutoscaling: false
kubectl apply -f ray-cluster-dual-workers.yaml
```

**Result**: Head pod becomes 1/1 Ready instead of 1/2, eliminating autoscaler issues entirely.
