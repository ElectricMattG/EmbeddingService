"""
Ray Serve deployment wrapper for EmbeddingService.

This module wraps the existing FastAPI application with Ray Serve
decorators to enable distributed deployment with autoscaling and
load balancing. No changes to the existing application logic required.
"""

from ray import serve

# Import the existing FastAPI app
from src.service.main import app


@serve.deployment(
    name="embedding-service",
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 0.2,  # Allow 2-5 replicas per GPU depending on model size
    },
    # Increase per-replica concurrency to tolerate larger bursts without Serve overload rejections.
    # Default is 5; previously 64; now 100 for additional headroom. Tune based on latency SLOs.
    max_ongoing_requests=100,
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 16,
        "target_num_ongoing_requests_per_replica": 4,
    },
)
@serve.ingress(app)
class EmbeddingServiceDeployment:
    """
    Ray Serve deployment that wraps the FastAPI application.
    
    The @serve.ingress decorator makes all FastAPI routes available
    through Ray Serve, preserving all existing endpoints, middleware,
    and business logic without modification.
    """
    pass


# Deployment binding for Ray Serve
deployment = EmbeddingServiceDeployment.bind()
