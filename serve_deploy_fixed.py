#!/usr/bin/env python3
"""
Fixed Ray Serve deployment script for EmbeddingService.
This version uses absolute imports to avoid module path issues.
"""

import sys

import ray
from ray import serve

# Ensure the app directory is in Python path
sys.path.append('/app')

# Initialize Ray
ray.init(address="auto")

# Import using absolute path from app directory
from src.service.main import app


@serve.deployment(
    name="embedding-service",
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 16,
        "target_ongoing_requests": 4,
    },
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0.5,
    }
)
class EmbeddingServiceDeployment:
    def __init__(self, max_batch: int = 512):
        """Initialize the embedding service with the given configuration."""
        print(f"🚀 Initializing EmbeddingService with max_batch={max_batch}")

    async def __call__(self, request):
        """Handle incoming requests by forwarding to the FastAPI app."""
        return await app(request)

# Deploy the application
deployment = EmbeddingServiceDeployment.bind(max_batch=512)

handle = serve.run(
    deployment,
    name="embedding-service",
    route_prefix="/",
)

print("✅ EmbeddingService deployed to Ray Serve successfully!")
print(f"📊 Cluster resources: {ray.cluster_resources()}")
print(f"🚀 Service available at: http://<cluster-ip>:8000")
print(f"🔍 Serve status: {serve.status()}")
