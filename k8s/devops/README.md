# DevOps Tools Deployment

This directory contains Kubernetes manifests for deploying DevOps tools (ArgoCD and Jenkins) on a dedicated node pool.

## Infrastructure

### DevOps Node Pool
- **Name**: devopspool
- **VM Size**: Standard_D2s_v3 (2 vCPU, 8GB RAM)
- **Node Count**: 1
- **Labels**: `workload=devops`
- **Taints**: `workload=devops:NoSchedule`

All DevOps tools are configured with node selectors and tolerations to run exclusively on this pool.

## ArgoCD

ArgoCD is a declarative, GitOps continuous delivery tool for Kubernetes.

### Components
- **argocd-server**: Web UI and API server
- **argocd-repo-server**: Repository service for Git operations
- **argocd-application-controller**: Application controller that monitors apps

### Installation

```bash
# Create ArgoCD namespace and deploy
kubectl apply -f k8s/devops/argocd-install.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=argocd-server -n argocd --timeout=300s

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### Access

**URL**: https://argocd.rapporteur.ai

**Default Credentials**:
- Username: `admin`
- Password: Retrieved from the secret (see installation steps)

**CLI Login**:
```bash
# Install ArgoCD CLI
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd

# Login
argocd login argocd.rapporteur.ai --username admin --password <password>
```

### Adding Git Repositories

**Via UI**:
1. Navigate to Settings → Repositories
2. Click "Connect Repo"
3. Enter repository URL and credentials (if private)

**Via CLI**:
```bash
# Public repository
argocd repo add https://github.com/ElectricMattG/EmbeddingService.git

# Private repository with SSH
argocd repo add git@github.com:ElectricMattG/EmbeddingService.git \
  --ssh-private-key-path ~/.ssh/id_rsa

# Private repository with HTTPS
argocd repo add https://github.com/ElectricMattG/EmbeddingService.git \
  --username <username> \
  --password <token>
```

### Creating Applications

**Via UI**:
1. Navigate to Applications
2. Click "New App"
3. Fill in details (name, project, source, destination)
4. Click "Create"

**Via CLI**:
```bash
argocd app create embedding-service \
  --repo https://github.com/ElectricMattG/EmbeddingService.git \
  --path k8s \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default
```

**Via Manifest**:
```bash
kubectl apply -f k8s/devops/argocd-application-sample.yaml
```

### Sync Applications

```bash
# Manual sync
argocd app sync embedding-service

# Enable auto-sync
argocd app set embedding-service --sync-policy automated

# View sync status
argocd app get embedding-service
```

## Jenkins

Jenkins is an automation server for building, testing, and deploying applications.

### Installation

```bash
# Deploy Jenkins
kubectl apply -f k8s/devops/jenkins-deployment.yaml

# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=jenkins -n jenkins --timeout=300s

# Get initial admin password
kubectl exec -n jenkins deployment/jenkins -- cat /var/jenkins_home/secrets/initialAdminPassword
```

### Access

**URL**: https://jenkins.rapporteur.ai

**Initial Setup**:
1. Use the password from installation steps
2. Install suggested plugins
3. Create admin user
4. Configure instance URL: https://jenkins.rapporteur.ai

### Kubernetes Plugin Configuration

Jenkins is pre-configured with a service account that has cluster access.

1. Install "Kubernetes" plugin (if not already installed)
2. Go to Manage Jenkins → Configure System
3. Add Kubernetes Cloud:
   - Name: `kubernetes`
   - Kubernetes URL: `https://kubernetes.default.svc`
   - Kubernetes Namespace: `jenkins`
   - Credentials: Use service account token
   - Jenkins URL: `http://jenkins.jenkins.svc.cluster.local:8080`

### Sample Pipeline with Kubernetes

```groovy
pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: docker
    image: docker:latest
    command:
    - cat
    tty: true
    volumeMounts:
    - mountPath: /var/run/docker.sock
      name: docker-sock
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
'''
        }
    }
    stages {
        stage('Build') {
            steps {
                container('docker') {
                    sh 'docker build -t embedding-service:${BUILD_NUMBER} .'
                }
            }
        }
        stage('Push') {
            steps {
                container('docker') {
                    sh 'docker push ragserviceneacr6881.azurecr.io/embedding-service:${BUILD_NUMBER}'
                }
            }
        }
    }
}
```

### Connecting to Git Repositories

1. Go to Manage Jenkins → Manage Credentials
2. Add credentials for your Git provider:
   - **GitHub**: Personal Access Token
   - **GitLab**: Deploy Token
   - **Bitbucket**: App Password

3. Create a new Pipeline job:
   - Pipeline from SCM
   - SCM: Git
   - Repository URL: Your repo URL
   - Credentials: Select created credentials
   - Branch: `*/mvp_dec_2025`

## Integration: ArgoCD + Jenkins

### CI/CD Workflow

1. **Jenkins**: Build and push Docker images
2. **Jenkins**: Update image tag in Git repository
3. **ArgoCD**: Detect Git changes and deploy to cluster

### Example Jenkins Pipeline for GitOps

```groovy
pipeline {
    agent any

    environment {
        GIT_REPO = 'https://github.com/ElectricMattG/EmbeddingService.git'
        IMAGE_TAG = "${BUILD_NUMBER}"
        ACR_REGISTRY = 'ragserviceneacr6881.azurecr.io'
    }

    stages {
        stage('Build') {
            steps {
                sh '''
                    echo "${IMAGE_TAG}" > version
                    docker build -t ${ACR_REGISTRY}/embedding-service:${IMAGE_TAG} .
                '''
            }
        }

        stage('Push') {
            steps {
                sh 'docker push ${ACR_REGISTRY}/embedding-service:${IMAGE_TAG}'
            }
        }

        stage('Update Manifests') {
            steps {
                script {
                    sh '''
                        git clone ${GIT_REPO} repo
                        cd repo
                        sed -i "s|image: ${ACR_REGISTRY}/embedding-service:.*|image: ${ACR_REGISTRY}/embedding-service:${IMAGE_TAG}|g" k8s/rayservice-aks-ne.yaml
                        git add k8s/rayservice-aks-ne.yaml
                        git commit -m "Update image to ${IMAGE_TAG}"
                        git push origin mvp_dec_2025
                    '''
                }
            }
        }

        stage('Trigger ArgoCD Sync') {
            steps {
                sh 'argocd app sync embedding-service --force'
            }
        }
    }
}
```

## DNS Configuration

Add A records to your DNS zone:

```bash
# Get ingress IP
kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Add DNS records (via Azure Portal or CLI)
az network dns record-set a add-record \
  --resource-group rapporteur-ne-rg \
  --zone-name rapporteur.ai \
  --record-set-name argocd \
  --ipv4-address <INGRESS_IP>

az network dns record-set a add-record \
  --resource-group rapporteur-ne-rg \
  --zone-name rapporteur.ai \
  --record-set-name jenkins \
  --ipv4-address <INGRESS_IP>
```

## Monitoring

### Check DevOps Node Pool

```bash
# List nodes
kubectl get nodes -l workload=devops

# Check node resources
kubectl top nodes -l workload=devops
```

### Check ArgoCD Status

```bash
# List applications
argocd app list

# Get application details
argocd app get embedding-service

# View application logs
kubectl logs -n argocd -l app=argocd-server --tail=100 -f
```

### Check Jenkins Status

```bash
# Get Jenkins pod
kubectl get pods -n jenkins

# View Jenkins logs
kubectl logs -n jenkins -l app=jenkins --tail=100 -f

# Check PVC
kubectl get pvc -n jenkins
```

## Scaling

### Scale DevOps Node Pool

```bash
# Scale up
az aks nodepool scale \
  --resource-group rapporteur-ne-rg \
  --cluster-name embedding-aks-ne \
  --name devopspool \
  --node-count 2

# Scale down
az aks nodepool scale \
  --resource-group rapporteur-ne-rg \
  --cluster-name embedding-aks-ne \
  --name devopspool \
  --node-count 1
```

### Scale Jenkins (if needed)

```bash
# Increase resources
kubectl patch deployment jenkins -n jenkins -p '{"spec":{"template":{"spec":{"containers":[{"name":"jenkins","resources":{"requests":{"cpu":"1000m","memory":"2Gi"},"limits":{"cpu":"2000m","memory":"4Gi"}}}]}}}}'
```

## Security Best Practices

1. **Change Default Passwords**: Update ArgoCD admin password immediately
2. **Enable RBAC**: Configure role-based access control for both tools
3. **Use Git SSH Keys**: Prefer SSH over HTTPS for Git access
4. **Restrict Ingress**: Add IP whitelisting if needed
5. **Enable 2FA**: Configure multi-factor authentication for production
6. **Scan Images**: Use tools like Trivy to scan container images
7. **Secret Management**: Use Azure Key Vault for sensitive data

## Troubleshooting

### ArgoCD Not Syncing

```bash
# Check application status
argocd app get embedding-service

# View sync logs
kubectl logs -n argocd -l app=argocd-application-controller

# Force refresh
argocd app get embedding-service --refresh
```

### Jenkins Build Failing

```bash
# Check pod logs
kubectl logs -n jenkins -l app=jenkins

# Check PVC
kubectl describe pvc jenkins-pvc -n jenkins

# Restart Jenkins
kubectl rollout restart deployment/jenkins -n jenkins
```

### Pods Not Scheduling on DevOps Pool

```bash
# Check node status
kubectl describe node <devops-node-name>

# Verify taints and tolerations
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints

# Check pod events
kubectl describe pod <pod-name> -n <namespace>
```

## Cleanup

```bash
# Delete Jenkins
kubectl delete -f k8s/devops/jenkins-deployment.yaml

# Delete ArgoCD
kubectl delete -f k8s/devops/argocd-install.yaml

# Delete DevOps node pool
az aks nodepool delete \
  --resource-group rapporteur-ne-rg \
  --cluster-name embedding-aks-ne \
  --name devopspool
```
