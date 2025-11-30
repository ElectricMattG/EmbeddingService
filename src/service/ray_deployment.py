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
        "num_cpus": 1,
        "num_gpus": 0.2,  # Allow 2-5 replicas per GPU depending on model size
    },
    # Increased per-replica concurrency to coordinate with optimized semaphore limits.
    # Supports higher concurrency for 2-worker deployment with increased app-level limits.
    max_ongoing_requests=200,  # Increased from 100 to coordinate with doubled semaphores
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 16,
        "target_num_ongoing_requests_per_replica": 8,  # Increased from 4 for higher baseline
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
