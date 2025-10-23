## Operations notes for EmbeddingService v4.0.1

This document captures operational guidance and the production-hardening changes introduced in v4.0.1.

### What changed in v4.0.1

- Fixed intermittent 500s under bursty segmentation workloads by serializing Hugging Face tokenizer access with a thread lock.
- Pinned container image across manifests and Helm values to `192.168.1.3:5000/embedding-service:v4.0.1` with pullPolicy `Always`.
- Switched worker readiness probes to use the node-local raylet health endpoint instead of Serve HTTP on workers, improving readiness determinism:
  - Readiness probe: `wget --tries 1 -T 2 -q -O- http://localhost:52365/api/local_raylet_healthz | grep success`
- Head node start parameters updated:
  - `num-cpus: 0` (don’t schedule work on head)
  - `dashboard-host: 0.0.0.0` (dashboard accessible in-cluster)

### Deploying

1. Build and push image (local registry shown):
   - Tag: `192.168.1.3:5000/embedding-service:v4.0.1`
2. Apply RayService manifest `ray-cluster-dual-workers.yaml` (pinned to v4.0.1) or use Helm (see below).
3. During rolling updates, proactively delete old GPU worker pods if GPUs are saturated to free resources for the new revision.

### Kubernetes and Ray

- Ray version: 2.39.0
- Serve HTTP listens on port 8000; dashboard on 8265 (head only).
- Serve location may be head-only depending on placement; worker readiness is independent via raylet healthz.

### Ingress (NGINX)

- Use increased timeouts, limited retries (`proxy-next-upstream`), and larger proxy buffers for large JSON payloads.
- Enable gzip for `application/json`.

### Health and readiness

- App health endpoints (FastAPI): `/health`, `/health/live`, `/health/ready`, `/health/status`, `/health/metrics`.
- Worker readiness (K8s): raylet healthz (see probe above).

### Troubleshooting checklist

- Bursts returning 500s with “Already borrowed”/tokenizer errors: ensure v4.0.1+ image is deployed (tokenizer lock fix).
- Workers stuck NotReady: confirm raylet healthz succeeds inside the pod; avoid relying on Serve HTTP on workers.
- Revision rollouts stalled due to GPU pressure: delete old worker pods to free GPUs; re-apply the RayService.
- Dashboard access: port-forward `8265` to inspect cluster; verify GCS and node health; check `/tmp/ray/session_latest/logs/` in pods.

### Quick log locations (inside pods)

- `/tmp/ray/session_latest/logs/raylet.out`
- `/tmp/ray/session_latest/logs/dashboard_agent.log`
- `/tmp/ray/session_latest/logs/runtime_env_agent.out`

### Validation results (sample)

- 200 requests @ P10 (segmented + segments): 100% 200s, mean ~0.0676s
- 200 requests @ P10 (no segmentation): 100% 200s, mean ~0.0461s
# Operational Notes (v4.0.1)

- Image: `192.168.1.3:5000/embedding-service:v4.0.1` (pinned)
- Ray/Ray Serve: `2.39.0`
- Cluster: RayService with 1 head + 2 GPU workers

## Readiness and health

- Worker readiness probes check the local raylet only (robust when Serve HTTP proxy isn’t on every worker):
  - `wget --tries 1 -T 2 -q -O- http://localhost:52365/api/local_raylet_healthz | grep success`
- Head exposes:
  - Dashboard: 8265
  - Serve HTTP: 8000

## Validated ingress performance (P=10)

- Segmentation + include_segment_embeddings: 200/200 OK, mean ~0.0676s
- No segmentation: 200/200 OK, mean ~0.0461s

## Dashboard access

- Port-forward:
  - `kubectl port-forward --address 0.0.0.0 svc/ray-cluster-simple-head-svc 8265:8265`
- Open: http://<host>:8265

## Manifests pinned to v4.0.1

- Kubernetes: `ray-cluster-dual-workers.yaml` (head/worker images)
- Helm: `helm/values-embedding-service.yaml` (repository/tag/pullPolicy)

## Redeploy guidance

- Follow `docs/forceful-redeploy.md` when replacing images or versions:
  - Delete RayService and force-delete lingering pods if needed
  - Verify no dangling GPU processes (`nvidia-smi`)
  - Re-apply manifest and wait for Ready
  - Run `/health` and a brief burst test to confirm
