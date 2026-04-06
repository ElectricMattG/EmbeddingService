SHELL := /bin/bash
VIRTUALENV_NAME := $(shell basename $(CURDIR))

PIP_COMPILE_ARGS = -v -U --no-emit-index-url --no-emit-trusted-host --strip-extras --resolver=backtracking

PYTHON_VERSION = 3.12.1 # keep minor versions syncd

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
BOLD := \033[1m
RESET := \033[0m

.PHONY: _python_version
_python_version:
	@if [ -z $(PYTHON_VERSION) ]; then \
		echo "$(RED)$(BOLD)This is the python verion $(PYTHON_VERSION)$(RESET)"; \
		exit 1; \
	fi

.PHONY: pyenv
pyenv: | _python_version # create a python virtual env
	@echo "$(YELLOW)$(BOLD)Creating a virtual environment$(RESET)"
	@if ! pyenv versions --bare | grep -q $(PYTHON_VERSION); then \
		echo "Python version $(PYTHON_VERSION) is not installed."; \
		echo "Installing this version of Python..."; \
		pyenv install -s $(PYTHON_VERSION); \
	fi
	@if ! pyenv versions --bare | grep -q $(VIRTUALENV_NAME); then \
		pyenv virtualenv $(PYTHON_VERSION) $(VIRTUALENV_NAME); else \
		echo "Virtual environment $(VIRTUALENV_NAME) already exists."; \
	fi
	@pyenv local $(VIRTUALENV_NAME)
	@pip install --upgrade pip

.PHONY: pre-commit
pre-commit:
	@pip install -q pre-commit
	@pre-commit install

.PHONY: check
check: | pre-commit
	@echo "$(YELLOW)$(BOLD)Running checks (pre-commit)$(RESET)"
	pre-commit run --all-files

.PHONY: pip-tools
pip-tools:
	@echo "$(YELLOW)$(BOLD)Installing pip-tools (pre-commit)$(RESET)"
	@pip install -q pip-tools

%.txt: %.in | pip-tools # Compile specified requirements (pip-compile).
	@echo "$(YELLOW)$(BOLD)Compiling $*.txt$(RESET)"
	CUSTOM_COMPILE_COMMAND="make update" pip-compile $(PIP_COMPILE_ARGS) $*.in

.PHONY: requirements
requirements: requirements.in dev-requirements.in | pip-tools
	@echo "$(YELLOW)$(BOLD)Compiling requirements$(RESET)"
	CUSTOM_COMPILE_COMMAND="make update" pip-compile $(PIP_COMPILE_ARGS) requirements.in --output-file requirements.txt
	CUSTOM_COMPILE_COMMAND="make update" pip-compile $(PIP_COMPILE_ARGS) dev-requirements.in --output-file dev-requirements.txt

.PHONY: sync
sync: requirements.txt dev-requirements.txt | pip-tools
	@echo "$(YELLOW)$(BOLD)Syncing dependencies$(RESET)"
	pip-sync requirements.txt dev-requirements.txt

.PHONY: install
install: requirements.txt dev-requirements.txt
	@echo "$(YELLOW)$(BOLD)Installing dependencies$(RESET)"
	pip install \
	-r requirements.txt -r dev-requirements.txt
	pip install -e .

.PHONY: test
test: | install
	@echo "$(YELLOW)$(BOLD)Running tests$(RESET)"
	pytest

# Docker configuration
REGISTRY ?= 192.168.1.3:5000
IMAGE_NAME ?= embedding-service
VERSION_FILE ?= version
VERSION ?= $(shell cat $(VERSION_FILE) 2>/dev/null || echo "latest")
FULL_IMAGE_TAG = $(REGISTRY)/$(IMAGE_NAME):$(VERSION)

# Kubernetes/Ray configuration
SERVICE_NAME ?= embedding-service
RAYSERVICE_FILE ?= k8s/rayservice-local.yaml
NAMESPACE ?= default
NODE_IP ?= $(shell ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($$i=="src") {print $$(i+1); exit}}')

#######################
# Docker Build & Push #
#######################

.PHONY: docker-build
docker-build:
	@echo "$(YELLOW)$(BOLD)Building Docker image: $(FULL_IMAGE_TAG)$(RESET)"
	docker build -t $(FULL_IMAGE_TAG) .
	docker tag $(FULL_IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):latest

.PHONY: docker-push
docker-push:
	@echo "$(YELLOW)$(BOLD)Pushing Docker image: $(FULL_IMAGE_TAG)$(RESET)"
	docker push $(FULL_IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
	@echo "$(YELLOW)$(BOLD)Cleaning up local images (available in registry)$(RESET)"
	docker rmi $(FULL_IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):latest --force 2>/dev/null || true

.PHONY: build-push
build-push: docker-build docker-push
	@echo "$(GREEN)$(BOLD)Successfully built and pushed: $(FULL_IMAGE_TAG)$(RESET)"

#######################
# Kubernetes Deploy   #
#######################

.PHONY: k8s-install-kuberay
k8s-install-kuberay:
	@echo "$(YELLOW)$(BOLD)Installing KubeRay operator$(RESET)"
	kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.4.2&timeout=90s" || true
	@echo "$(YELLOW)$(BOLD)Waiting for KubeRay operator to be ready$(RESET)"
	kubectl wait --for=condition=available --timeout=300s deployment/kuberay-operator -n kuberay-system

.PHONY: k8s-install-ingress
k8s-install-ingress:
	@echo "$(YELLOW)$(BOLD)Installing NGINX Ingress Controller$(RESET)"
	helm upgrade --install ingress-nginx ingress-nginx \
		--repo https://kubernetes.github.io/ingress-nginx \
		--namespace ingress-nginx --create-namespace \
		--set controller.hostNetwork=true \
		--set controller.kind=DaemonSet
	@echo "$(YELLOW)$(BOLD)Waiting for ingress controller to be ready$(RESET)"
	kubectl wait --namespace ingress-nginx \
		--for=condition=ready pod \
		--selector=app.kubernetes.io/component=controller \
		--timeout=300s

.PHONY: k8s-apply-storage
k8s-apply-storage:
	@echo "$(YELLOW)$(BOLD)Creating storage PVC (if not exists)$(RESET)"
	@kubectl get pvc ray-storage-pvc -n $(NAMESPACE) 2>/dev/null || \
	kubectl apply -f - <<EOF
	apiVersion: v1
	kind: PersistentVolumeClaim
	metadata:
	  name: ray-storage-pvc
	  namespace: $(NAMESPACE)
	spec:
	  accessModes:
	    - ReadWriteOnce
	  resources:
	    requests:
	      storage: 500Gi
	EOF

.PHONY: k8s-deploy-rayservice
k8s-deploy-rayservice:
	@echo "$(YELLOW)$(BOLD)Deploying RayService: $(RAYSERVICE_FILE)$(RESET)"
	kubectl apply -f $(RAYSERVICE_FILE) -n $(NAMESPACE)
	@echo "$(YELLOW)$(BOLD)Waiting for Ray pods (timeout 10m)$(RESET)"
	@for i in $$(seq 1 60); do \
	  CNT=$$(kubectl get pods -n $(NAMESPACE) -l ray.io/is-ray-node=yes -o name 2>/dev/null | wc -l); \
	  if [ "$$CNT" -gt 0 ]; then echo "Found $$CNT Ray pods"; break; fi; \
	  sleep 2; \
	done
	kubectl wait --for=condition=ready pod -n $(NAMESPACE) -l ray.io/is-ray-node=yes --timeout=600s
	@echo "$(GREEN)$(BOLD)RayService deployed successfully$(RESET)"
	kubectl get pods -n $(NAMESPACE) -l ray.io/is-ray-node=yes

.PHONY: k8s-apply-ingress
k8s-apply-ingress:
	@echo "$(YELLOW)$(BOLD)Applying ingress manifests$(RESET)"
	kubectl apply -f k8s/ingress/embedding-service-ingress.yaml
	kubectl apply -f k8s/ingress/ray-dashboard-ingress.yaml
	@echo "$(GREEN)$(BOLD)Ingress resources applied$(RESET)"

.PHONY: k8s-status
k8s-status:
	@echo "$(BLUE)$(BOLD)=== RayService Status ===$(RESET)"
	kubectl get rayservice -n $(NAMESPACE)
	@echo ""
	@echo "$(BLUE)$(BOLD)=== Ray Pods ===$(RESET)"
	kubectl get pods -n $(NAMESPACE) -l ray.io/is-ray-node=yes
	@echo ""
	@echo "$(BLUE)$(BOLD)=== Services ===$(RESET)"
	kubectl get svc -n $(NAMESPACE) | grep $(SERVICE_NAME)
	@echo ""
	@echo "$(BLUE)$(BOLD)=== Ingress ===$(RESET)"
	kubectl get ingress -n $(NAMESPACE)

.PHONY: k8s-test
k8s-test:
	@echo "$(YELLOW)$(BOLD)Testing API via ingress$(RESET)"
	@IP_DASH=$$(echo $(NODE_IP) | tr . -); \
	HOST=embeddings.$$IP_DASH.nip.io; \
	echo "Testing: http://$$HOST/health"; \
	curl -s http://$$HOST/health | jq .

.PHONY: k8s-delete
k8s-delete:
	@echo "$(YELLOW)$(BOLD)Deleting RayService$(RESET)"
	kubectl delete rayservice $(SERVICE_NAME) -n $(NAMESPACE) --ignore-not-found=true
	@echo "$(YELLOW)$(BOLD)Waiting for pods to terminate$(RESET)"
	@sleep 5
	@kubectl delete pods -n $(NAMESPACE) -l ray.io/is-ray-node=yes --force --grace-period=0 2>/dev/null || true

.PHONY: k8s-cleanup-all
k8s-cleanup-all:
	@echo "$(YELLOW)$(BOLD)Deleting ALL RayServices and Ray pods$(RESET)"
	kubectl delete rayservice --all -n $(NAMESPACE) --ignore-not-found=true
	@sleep 5
	kubectl delete pods --all -n $(NAMESPACE) --force --grace-period=0 2>/dev/null || true
	@echo "$(YELLOW)$(BOLD)Cleaning up Ray services$(RESET)"
	kubectl delete svc -n $(NAMESPACE) -l ray.io/cluster --ignore-not-found=true 2>/dev/null || true
	kubectl delete svc ray-dashboard-external -n $(NAMESPACE) --ignore-not-found=true 2>/dev/null || true
	@echo "$(GREEN)$(BOLD)✓ Cluster cleaned$(RESET)"

.PHONY: docker-cleanup
docker-cleanup:
	@echo "$(YELLOW)$(BOLD)Cleaning Docker resources$(RESET)"
	docker builder prune -af
	docker system prune -f
	@echo "$(GREEN)$(BOLD)✓ Docker cleanup complete$(RESET)"
	@df -h / | grep -E "Filesystem|/$"

.PHONY: full-cleanup
full-cleanup: k8s-cleanup-all docker-cleanup
	@echo "$(GREEN)$(BOLD)✓ Full cleanup complete$(RESET)"

#######################
# Quick Start         #
#######################

.PHONY: quickstart
quickstart: k8s-install-kuberay k8s-install-ingress k8s-apply-storage k8s-deploy-rayservice k8s-apply-ingress k8s-test
	@echo "$(GREEN)$(BOLD)✓ Quickstart complete!$(RESET)"
	@echo ""
	@echo "$(BLUE)$(BOLD)Access your service:$(RESET)"
	@IP_DASH=$$(echo $(NODE_IP) | tr . -); \
	echo "  API:       http://embeddings.$$IP_DASH.nip.io"; \
	echo "  Dashboard: http://dashboard.$$IP_DASH.nip.io"

.PHONY: redeploy
redeploy: k8s-delete k8s-deploy-rayservice k8s-test
	@echo "$(GREEN)$(BOLD)✓ Redeploy complete!$(RESET)"
