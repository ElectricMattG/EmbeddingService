<H1>EmbeddingService</H1>

## Ingress (NGINX) configuration for burst resilience

For external access via NGINX Ingress with better resilience under bursts and large payloads, apply the manifest at `helm/nginx-ingress/embedding-service-ingress.yaml` and update the host to your DNS (or nip.io host). Key settings:

- proxy-next-upstream with limited retries for transient 5xx
- increased proxy buffer sizes for large JSON responses
- gzip enabled for application/json
- extended proxy timeouts for long-running requests

Example host override:

- Set `spec.rules[0].host` to your ingress host (e.g., `embeddings.192-168-1-3.nip.io`).

Note: The service name defaults to `ray-cluster-simple-serve-svc` on port 8000; adjust if your service name differs.

## App-level compression

The FastAPI app enables GZip compression for responses larger than 1KB. This reduces upstream load and bandwidth, particularly when returning segment embeddings.
