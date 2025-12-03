#!/usr/bin/env python3
"""
Ray Serve deployment script for EmbeddingService.

This script deploys the EmbeddingService application to Ray Serve
with the configured autoscaling and GPU allocation settings.
"""

import ray
from ray import serve

# Initialize Ray
ray.init(address="auto")

# Import the deployment
from src.service.ray_deployment import deployment

# Deploy the application
# Note: Ray Serve 2.39 binds to 0.0.0.0 by default in Kubernetes
handle = serve.run(
    deployment,
    name="embedding-service",
    route_prefix="/",
)

print("✅ EmbeddingService deployed to Ray Serve successfully!")
print(f"📊 Cluster resources: {ray.cluster_resources()}")
print(f"🚀 Service available at: http://<cluster-ip>:8000")
print(f"🔍 Serve status: {serve.status()}")
