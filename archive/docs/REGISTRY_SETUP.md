# Local Docker Registry Setup Guide

## Overview

This guide covers setting up and using a local Docker registry to overcome Kubernetes disk-pressure constraints when deploying large container images. The registry stores images on an external volume with ample storage space, enabling incremental layer pulls instead of full image imports.

## Problem Statement

**Issue**: Kubernetes pods fail to schedule with `disk-pressure` taints even when overall disk usage is only 65%.

**Root Cause**: Ephemeral storage exhaustion during containerd image layer extraction. When Kubernetes pulls a large image (7.96GB embedding-service), containerd needs temporary space to extract and process layers before storing them in content-addressable storage. This temporary extraction can require significant ephemeral storage beyond the final image size.

**Solution**: Deploy a local Docker registry on a volume with ample free space (`/mnt/workspace_2` with 2.3TB available). Configure containerd to pull from this local registry, enabling incremental layer-by-layer pulls that minimize ephemeral storage usage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Development Workflow                                         │
│                                                              │
│  1. Build Image                                             │
│     docker build -t embedding-service:ray-latest .          │
│                                                              │
│  2. Tag for Local Registry                                  │
│     docker tag embedding-service:ray-latest \               │
│                192.168.1.3:5000/embedding-service:ray-latest│
│                                                              │
│  3. Push to Local Registry                                  │
│     docker push 192.168.1.3:5000/embedding-service:ray-latest│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Local Docker Registry (registry:2)                          │
│                                                              │
│  • Host: 192.168.1.3:5000                                   │
│  • Protocol: HTTP (insecure)                                │
│  • Storage: /mnt/workspace_2/docker-registry                │
│  • Space: 2.3TB available (exFAT filesystem)                │
│  • Restart Policy: always                                   │
│                                                              │
│  Storage Structure:                                         │
│  /mnt/workspace_2/docker-registry/                          │
│    └── docker/registry/v2/                                  │
│        ├── repositories/                                    │
│        │   └── embedding-service/                           │
│        │       ├── _manifests/                              │
│        │       │   └── tags/ray-latest/                     │
│        │       └── _layers/                                 │
│        │           └── sha256/                              │
│        │               └── [layer blobs]                    │
│        └── blobs/                                           │
│            └── sha256/                                      │
│                └── [content-addressable layer data]        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Containerd Configuration (/etc/containerd/config.toml)     │
│                                                              │
│  [plugins."io.containerd.grpc.v1.cri".registry]            │
│    [plugins."...".registry.mirrors."192.168.1.3:5000"]     │
│      endpoint = ["http://192.168.1.3:5000"]                │
│                                                              │
│    [plugins."...".registry.configs."192.168.1.3:5000".tls]│
│      insecure_skip_verify = true                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Kubernetes Image Pull                                       │
│                                                              │
│  Pod Spec:                                                  │
│    image: 192.168.1.3:5000/embedding-service:ray-latest    │
│    imagePullPolicy: Always                                  │
│                                                              │
│  Pull Process (Incremental):                                │
│  1. Containerd requests manifest from registry             │
│  2. Registry returns layer list (12 layers)                │
│  3. Containerd pulls each layer individually               │
│  4. Layers extracted directly to content store             │
│  5. No large temporary extraction (minimal ephemeral usage)│
└─────────────────────────────────────────────────────────────┘
```

---

## Initial Setup

### 1. Prepare Storage Location

Choose a volume with ample free space (preferably 100GB+ for multiple large images):

```bash
# Check available space
df -h /mnt/workspace_2
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sdb1       3.7T  1.4T  2.3T  39% /mnt/workspace_2

# Create registry storage directory
mkdir -p /mnt/workspace_2/docker-registry

# Verify ownership and permissions
ls -ld /mnt/workspace_2/docker-registry
# drwxr-xr-x 2 electric electric 131072 Oct 22 16:03 /mnt/workspace_2/docker-registry
```

**Storage Recommendations**:
- **Minimum**: 50GB (for 5-10 medium-sized images)
- **Recommended**: 100GB+ (for multiple large ML/AI images)
- **Filesystem**: Any (ext4, xfs, exFAT all work)
- **Permissions**: Read/write for user running Docker daemon

### 2. Deploy Docker Registry

```bash
# Pull official registry image
docker pull registry:2

# Start registry container
docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name local-registry \
  -v /mnt/workspace_2/docker-registry:/var/lib/registry \
  registry:2

# Verify registry is running
docker ps | grep local-registry
# CONTAINER ID   IMAGE        COMMAND                  CREATED         STATUS         PORTS                    NAMES
# 5f37a2ba1391   registry:2   "/entrypoint.sh /etc…"   2 hours ago     Up 2 hours     0.0.0.0:5000->5000/tcp   local-registry

# Check registry logs
docker logs local-registry
# time="2025-10-22T14:00:00Z" level=info msg="listening on [::]:5000"
```

**Registry Options Explained**:
- `-d`: Run in detached mode (background)
- `-p 5000:5000`: Expose port 5000 on host
- `--restart=always`: Auto-restart on system reboot
- `--name local-registry`: Container name for easy reference
- `-v /mnt/workspace_2/docker-registry:/var/lib/registry`: Mount external storage

### 3. Test Registry Connectivity

```bash
# Get node IP address
NODE_IP=$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
echo "Node IP: $NODE_IP"
# Node IP: 192.168.1.3

# Test registry API (empty catalog initially)
curl -s http://$NODE_IP:5000/v2/_catalog
# {"repositories":[]}

# Check registry health
curl -s http://$NODE_IP:5000/v2/
# {}  (empty JSON means healthy)
```

### 4. Configure Containerd for Insecure Registry

Kubernetes uses containerd as the container runtime. By default, containerd only trusts HTTPS registries. We need to configure it to trust our HTTP registry.

```bash
# Backup existing config
sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.backup

# Edit containerd config
sudo nano /etc/containerd/config.toml
```

**Add this configuration block** (replace `192.168.1.3` with your node IP):

```toml
# Registry mirror configuration
[plugins."io.containerd.grpc.v1.cri".registry]
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."192.168.1.3:5000"]
      endpoint = ["http://192.168.1.3:5000"]

  [plugins."io.containerd.grpc.v1.cri".registry.configs]
    [plugins."io.containerd.grpc.v1.cri".registry.configs."192.168.1.3:5000"]
      [plugins."io.containerd.grpc.v1.cri".registry.configs."192.168.1.3:5000".tls]
        insecure_skip_verify = true
```

**Configuration Explained**:
- `registry.mirrors`: Tell containerd to use our registry for images matching `192.168.1.3:5000/*`
- `endpoint`: HTTP URL of registry
- `insecure_skip_verify`: Skip TLS verification (required for HTTP registries)

**Restart containerd** to apply changes:

```bash
# Restart containerd service
sudo systemctl restart containerd

# Verify containerd is running
sudo systemctl status containerd
# ● containerd.service - containerd container runtime
#    Loaded: loaded (/lib/systemd/system/containerd.service; enabled)
#    Active: active (running)

# Check containerd can reach registry
sudo crictl --runtime-endpoint unix:///run/containerd/containerd.sock version
```

---

## Building and Pushing Images

### Standard Workflow

#### 1. Build Docker Image

```bash
# Navigate to project directory
cd /home/electric/workspace/EmbeddingService

# Build image with tag
docker build -t embedding-service:ray-latest .

# Verify build succeeded
docker images | grep embedding-service
# REPOSITORY          TAG          IMAGE ID       CREATED         SIZE
# embedding-service   ray-latest   a1b2c3d4e5f6   2 minutes ago   7.96GB
```

**Build Tips**:
- Use `.dockerignore` to exclude unnecessary files (logs, cache, __pycache__)
- Use `--progress=plain` for detailed build output during debugging
- Consider multi-stage builds to reduce final image size

#### 2. Tag for Local Registry

```bash
# Tag with registry hostname:port prefix
docker tag embedding-service:ray-latest 192.168.1.3:5000/embedding-service:ray-latest

# Verify tag
docker images | grep embedding-service
# REPOSITORY                                TAG          IMAGE ID       CREATED         SIZE
# embedding-service                         ray-latest   a1b2c3d4e5f6   5 minutes ago   7.96GB
# 192.168.1.3:5000/embedding-service       ray-latest   a1b2c3d4e5f6   5 minutes ago   7.96GB
```

**Tagging Convention**:
```
[registry-host]:[port]/[repository]:[tag]
└─────────────────────┘ └──────────────────┘
     Required             Optional (default: latest)
```

#### 3. Push to Local Registry

```bash
# Push image to registry
docker push 192.168.1.3:5000/embedding-service:ray-latest

# Output shows layer pushes:
# The push refers to repository [192.168.1.3:5000/embedding-service]
# ecfd70c4572c: Pushed
# 93d5d25eb619: Pushed
# 5307c833d99b: Pushed
# 54519a429305: Pushed
# 878514c37e6e: Pushed
# 7ea3bfc3aef6: Pushed
# db1804a987b0: Pushed
# ad95a3d7aac4: Pushed
# 7770fbb416af: Pushed
# acbbee3380a1: Pushed
# 16dffb6a6636: Pushed
# d7c97cb6f1fe: Pushed
# ray-latest: digest: sha256:0f61ed1375fa133063a789f63160cca6a2d6bcb4d0250352454fd81417d30451 size: 2826
```

**Push Process**:
1. Docker client sends manifest to registry
2. Registry checks which layers already exist (deduplication)
3. Only new/changed layers are uploaded
4. Registry stores layers in content-addressable storage
5. Manifest links layers together

#### 4. Verify Image in Registry

```bash
# List repositories in registry
curl -s http://192.168.1.3:5000/v2/_catalog | jq .
# {
#   "repositories": [
#     "embedding-service"
#   ]
# }

# List tags for repository
curl -s http://192.168.1.3:5000/v2/embedding-service/tags/list | jq .
# {
#   "name": "embedding-service",
#   "tags": [
#     "ray-latest"
#   ]
# }

# Get manifest (detailed image info)
curl -s http://192.168.1.3:5000/v2/embedding-service/manifests/ray-latest | jq '.config'
# {
#   "mediaType": "application/vnd.docker.container.image.v1+json",
#   "size": 7890,
#   "digest": "sha256:..."
# }
```

---

## Using Images in Kubernetes

### Update Kubernetes Manifests

Modify your pod/deployment manifests to use the local registry:

```yaml
# Before (won't work with large images):
spec:
  containers:
  - name: ray-head
    image: embedding-service:ray-latest
    imagePullPolicy: Never  # ❌ Causes issues

# After (works with local registry):
spec:
  containers:
  - name: ray-head
    image: 192.168.1.3:5000/embedding-service:ray-latest
    imagePullPolicy: Always  # ✅ Pulls from registry
```

**Example**: `ray-cluster-dual-workers.yaml`

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-cluster-simple
spec:
  headGroupSpec:
    template:
      spec:
        containers:
        - name: ray-head
          image: 192.168.1.3:5000/embedding-service:ray-latest
          imagePullPolicy: Always
          # ... rest of config

  workerGroupSpecs:
  - groupName: worker-gpu
    replicas: 2
    template:
      spec:
        containers:
        - name: ray-worker
          image: 192.168.1.3:5000/embedding-service:ray-latest
          imagePullPolicy: Always
          # ... rest of config
```

### Deploy and Verify

```bash
# Apply manifest
kubectl apply -f ray-cluster-dual-workers.yaml

# Watch pods come up
kubectl get pods -l ray.io/cluster=ray-cluster-simple -w

# Check image pull events
kubectl describe pod <pod-name> | grep -A 10 Events
# Events:
#   Type    Reason     Age   From               Message
#   ----    ------     ----  ----               -------
#   Normal  Scheduled  30s   default-scheduler  Successfully assigned default/ray-cluster-simple-head-btggr to server
#   Normal  Pulling    29s   kubelet            Pulling image "192.168.1.3:5000/embedding-service:ray-latest"
#   Normal  Pulled     20s   kubelet            Successfully pulled image "192.168.1.3:5000/embedding-service:ray-latest"
#   Normal  Created    20s   kubelet            Created container ray-head
#   Normal  Started    19s   kubelet            Started container ray-head
```

**Success Indicators**:
- No `ImagePullBackOff` or `ErrImagePull` errors
- `Pulled` event shows registry URL
- Pull time is faster than initial build (layers cached in containerd)

---

## Advanced Operations

### Managing Multiple Images

```bash
# Build and push multiple versions
docker build -t embedding-service:v1.0 .
docker tag embedding-service:v1.0 192.168.1.3:5000/embedding-service:v1.0
docker push 192.168.1.3:5000/embedding-service:v1.0

docker build -t embedding-service:v1.1 .
docker tag embedding-service:v1.1 192.168.1.3:5000/embedding-service:v1.1
docker push 192.168.1.3:5000/embedding-service:v1.1

# List all tags
curl -s http://192.168.1.3:5000/v2/embedding-service/tags/list | jq .
# {
#   "name": "embedding-service",
#   "tags": [
#     "ray-latest",
#     "v1.0",
#     "v1.1"
#   ]
# }
```

### Updating Existing Images

When you rebuild an image with the same tag:

```bash
# Rebuild with changes
docker build -t embedding-service:ray-latest .

# Re-tag (same tag name)
docker tag embedding-service:ray-latest 192.168.1.3:5000/embedding-service:ray-latest

# Push (only changed layers uploaded)
docker push 192.168.1.3:5000/embedding-service:ray-latest

# Force Kubernetes to pull updated image
kubectl rollout restart deployment <deployment-name>
# OR
kubectl delete pod <pod-name>  # For RayCluster, delete pods to trigger repull
```

**Layer Caching**:
- Registry deduplicates layers by content hash (SHA256)
- If layer unchanged, push is instant (already exists in registry)
- Only modified layers are uploaded
- Example: Changing application code (small layer) = fast push, changing base image = slow push

### Deleting Images

```bash
# Get image digest (required for deletion)
DIGEST=$(curl -I -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  http://192.168.1.3:5000/v2/embedding-service/manifests/ray-latest 2>&1 | \
  grep Docker-Content-Digest | awk '{print $2}' | tr -d '\r')

echo "Digest: $DIGEST"
# Digest: sha256:0f61ed1375fa133063a789f63160cca6a2d6bcb4d0250352454fd81417d30451

# Delete manifest
curl -X DELETE http://192.168.1.3:5000/v2/embedding-service/manifests/$DIGEST

# Run garbage collection to free storage
docker exec local-registry bin/registry garbage-collect /etc/docker/registry/config.yml

# Verify deletion
curl -s http://192.168.1.3:5000/v2/embedding-service/tags/list
```

### Registry Garbage Collection

The registry doesn't automatically delete layer blobs when manifests are removed. Run garbage collection periodically:

```bash
# Stop registry (required for GC)
docker stop local-registry

# Run garbage collection
docker run --rm \
  -v /mnt/workspace_2/docker-registry:/var/lib/registry \
  registry:2 bin/registry garbage-collect /etc/docker/registry/config.yml

# Restart registry
docker start local-registry

# Check freed space
du -sh /mnt/workspace_2/docker-registry
```

**When to Run GC**:
- After deleting multiple old images/tags
- When storage space is running low
- Schedule monthly via cron for automated cleanup

### Backup and Restore

#### Backup Registry

```bash
# Create backup directory
mkdir -p /mnt/workspace_2/registry-backups

# Backup registry data
sudo tar -czf /mnt/workspace_2/registry-backups/registry-backup-$(date +%Y%m%d).tar.gz \
  -C /mnt/workspace_2 docker-registry

# Verify backup
ls -lh /mnt/workspace_2/registry-backups/
# -rw-r--r-- 1 root root 3.2G Oct 22 16:00 registry-backup-20251022.tar.gz
```

#### Restore Registry

```bash
# Stop registry
docker stop local-registry
docker rm local-registry

# Restore from backup
sudo tar -xzf /mnt/workspace_2/registry-backups/registry-backup-20251022.tar.gz \
  -C /mnt/workspace_2

# Restart registry
docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name local-registry \
  -v /mnt/workspace_2/docker-registry:/var/lib/registry \
  registry:2

# Verify restored images
curl -s http://192.168.1.3:5000/v2/_catalog | jq .
```

---

## Troubleshooting

### Issue: "connection refused" when pushing

**Symptoms**:
```bash
docker push 192.168.1.3:5000/embedding-service:ray-latest
# Error response from daemon: Get "https://192.168.1.3:5000/v2/": http: server gave HTTP response to HTTPS client
```

**Cause**: Docker client trying to use HTTPS

**Solution**: Configure Docker daemon to allow insecure registry

```bash
sudo nano /etc/docker/daemon.json
```

Add:
```json
{
  "insecure-registries": ["192.168.1.3:5000"]
}
```

Restart Docker:
```bash
sudo systemctl restart docker
docker info | grep -A 5 "Insecure Registries"
```

### Issue: Pods stuck with "ImagePullBackOff"

**Symptoms**:
```bash
kubectl get pods
# NAME                                    READY   STATUS             RESTARTS   AGE
# ray-cluster-simple-head-xxxxx           0/1     ImagePullBackOff   0          5m
```

**Check**:
```bash
kubectl describe pod <pod-name> | grep -A 10 Events
# Failed to pull image "192.168.1.3:5000/embedding-service:ray-latest": rpc error: code = Unknown desc = failed to pull and unpack image
```

**Common Causes**:

1. **Containerd not configured**:
   ```bash
   # Verify containerd config
   sudo grep -A 10 "192.168.1.3:5000" /etc/containerd/config.toml
   # Should show registry mirror config

   # Restart if missing
   sudo systemctl restart containerd
   ```

2. **Registry not accessible from node**:
   ```bash
   # Test from node (not local machine)
   curl http://192.168.1.3:5000/v2/_catalog
   # If fails: check firewall, registry container status
   ```

3. **Image doesn't exist in registry**:
   ```bash
   # Verify image exists
   curl -s http://192.168.1.3:5000/v2/embedding-service/tags/list
   # Check tag name matches pod spec
   ```

### Issue: Registry out of space

**Symptoms**:
```bash
docker push 192.168.1.3:5000/embedding-service:ray-latest
# error pushing to registry: write /var/lib/registry/...: no space left on device
```

**Check Space**:
```bash
df -h /mnt/workspace_2
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sdb1       3.7T  3.6T  100G  98% /mnt/workspace_2  ← Almost full!
```

**Solutions**:

1. **Run garbage collection** (see Advanced Operations)
2. **Delete old images**:
   ```bash
   # List all images
   curl -s http://192.168.1.3:5000/v2/_catalog | jq -r '.repositories[]' | while read repo; do
     echo "=== $repo ==="
     curl -s http://192.168.1.3:5000/v2/$repo/tags/list | jq .
   done

   # Delete unused tags
   ```

3. **Move registry to larger volume**:
   ```bash
   # Stop registry
   docker stop local-registry

   # Move data
   sudo mv /mnt/workspace_2/docker-registry /mnt/larger_volume/docker-registry

   # Update and restart registry
   docker rm local-registry
   docker run -d -p 5000:5000 --restart=always --name local-registry \
     -v /mnt/larger_volume/docker-registry:/var/lib/registry \
     registry:2
   ```

### Issue: Slow image pulls

**Symptoms**: Pods take 5-10+ minutes to pull images

**Causes**:
1. **Large image size**: 7.96GB takes time
2. **Network bottleneck**: Local network congestion
3. **Disk I/O**: Storage volume is slow (e.g., USB drive)

**Optimizations**:

1. **Reduce image size**:
   ```dockerfile
   # Use multi-stage builds
   FROM python:3.12-slim as builder
   RUN pip install --no-cache-dir -r requirements.txt

   FROM python:3.12-slim
   COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
   ```

2. **Pre-pull images to nodes**:
   ```bash
   # Manually pull on each node
   sudo crictl pull 192.168.1.3:5000/embedding-service:ray-latest
   ```

3. **Use faster storage**: Move registry to SSD instead of HDD

---

## Best Practices

### Security

1. **Production**: Use HTTPS with TLS certificates
   ```bash
   # Generate self-signed cert
   openssl req -newkey rsa:4096 -nodes -sha256 \
     -keyout registry.key -x509 -days 365 -out registry.crt

   # Run registry with TLS
   docker run -d -p 5000:5000 --restart=always --name local-registry \
     -v /mnt/workspace_2/docker-registry:/var/lib/registry \
     -v $(pwd)/certs:/certs \
     -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/registry.crt \
     -e REGISTRY_HTTP_TLS_KEY=/certs/registry.key \
     registry:2
   ```

2. **Authentication**: Add basic auth for multi-user environments
   ```bash
   # Create password file
   docker run --entrypoint htpasswd httpd:2 -Bbn user password > htpasswd

   # Run registry with auth
   docker run -d -p 5000:5000 --restart=always --name local-registry \
     -v /mnt/workspace_2/docker-registry:/var/lib/registry \
     -v $(pwd)/auth:/auth \
     -e REGISTRY_AUTH=htpasswd \
     -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
     -e REGISTRY_AUTH_HTPASSWD_REALM="Registry Realm" \
     registry:2

   # Login before push
   docker login 192.168.1.3:5000
   ```

3. **Network isolation**: Bind registry to internal network only
   ```bash
   docker run -d -p 127.0.0.1:5000:5000 ...  # Only accessible locally
   ```

### Performance

1. **Use SSD for registry storage** (faster layer reads)
2. **Enable registry caching** for frequently pulled images
3. **Run registry on same node as Kubernetes** (eliminates network latency)

### Maintenance

1. **Monitor storage usage**:
   ```bash
   # Add to cron
   echo "0 2 * * * du -sh /mnt/workspace_2/docker-registry >> /var/log/registry-size.log" | crontab -
   ```

2. **Automate backups**:
   ```bash
   # Daily backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d)
   tar -czf /mnt/workspace_2/registry-backups/registry-$DATE.tar.gz \
     -C /mnt/workspace_2 docker-registry
   find /mnt/workspace_2/registry-backups -mtime +30 -delete  # Keep 30 days
   ```

3. **Schedule garbage collection**:
   ```bash
   # Monthly GC
   0 3 1 * * docker stop local-registry && \
     docker run --rm -v /mnt/workspace_2/docker-registry:/var/lib/registry \
     registry:2 bin/registry garbage-collect /etc/docker/registry/config.yml && \
     docker start local-registry
   ```

---

## Quick Reference

### Essential Commands

```bash
# Check registry status
docker ps | grep local-registry
curl http://192.168.1.3:5000/v2/_catalog

# Build, tag, push workflow
docker build -t myimage:latest .
docker tag myimage:latest 192.168.1.3:5000/myimage:latest
docker push 192.168.1.3:5000/myimage:latest

# List repositories and tags
curl -s http://192.168.1.3:5000/v2/_catalog | jq .
curl -s http://192.168.1.3:5000/v2/myimage/tags/list | jq .

# Restart registry
docker restart local-registry

# View registry logs
docker logs -f local-registry

# Check storage usage
du -sh /mnt/workspace_2/docker-registry
```

### Configuration Files

**Docker daemon** (`/etc/docker/daemon.json`):
```json
{
  "insecure-registries": ["192.168.1.3:5000"]
}
```

**Containerd** (`/etc/containerd/config.toml`):
```toml
[plugins."io.containerd.grpc.v1.cri".registry.mirrors."192.168.1.3:5000"]
  endpoint = ["http://192.168.1.3:5000"]

[plugins."io.containerd.grpc.v1.cri".registry.configs."192.168.1.3:5000".tls]
  insecure_skip_verify = true
```

---

## Summary

This local Docker registry setup solves Kubernetes disk-pressure issues by:

1. **Storing images externally**: Uses `/mnt/workspace_2` (2.3TB) instead of limited root filesystem
2. **Incremental pulls**: Containerd pulls layers individually, minimizing ephemeral storage usage
3. **Layer deduplication**: Shared base layers stored once, saving space
4. **Fast local access**: No external registry dependencies or rate limits

**Key Takeaways**:
- Always tag images with full registry path: `192.168.1.3:5000/image:tag`
- Use `imagePullPolicy: Always` in Kubernetes manifests
- Configure containerd with `insecure_skip_verify` for HTTP registries
- Monitor storage usage and run garbage collection periodically
- Backup registry data before major changes

**Next Steps**:
- See `README_RAY_K8S.md` for full Ray Serve deployment guide
- Configure TLS for production deployments
- Set up automated backups and monitoring
