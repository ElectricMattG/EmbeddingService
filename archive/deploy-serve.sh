#!/bin/bash
# Deploy Ray Serve application to the Ray cluster

set -e

CLUSTER_NAME="ray-cluster-simple"
NAMESPACE="default"

echo "Deploying Ray Serve application to cluster: ${CLUSTER_NAME}"

# Wait for Ray cluster to be ready
echo "Waiting for Ray cluster pods to be ready..."
kubectl wait --for=condition=Ready pod -l ray.io/cluster=${CLUSTER_NAME} --timeout=300s -n ${NAMESPACE}

# Get the head pod name
HEAD_POD=$(kubectl get pods -n ${NAMESPACE} -l ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
echo "Head pod: ${HEAD_POD}"

# Deploy the Serve application
echo "Deploying Serve application..."
kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- python -c "
import ray
from ray import serve

# Connect to Ray cluster
ray.init(address='auto')

# Deploy the application
serve.run(
    target='src.service.ray_deployment:deployment',
    name='embedding-service',
    route_prefix='/',
)

print('Ray Serve application deployed successfully!')
"

echo ""
echo "Deployment complete!"
echo ""
echo "To access the service:"
echo "  kubectl port-forward -n ${NAMESPACE} svc/${CLUSTER_NAME}-head-svc 8000:8000"
echo ""
echo "To access Ray Dashboard:"
echo "  kubectl port-forward -n ${NAMESPACE} svc/${CLUSTER_NAME}-head-svc 8265:8265"
echo ""
echo "Test the service:"
echo "  curl http://localhost:8000/health/ready"
