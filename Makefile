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


.PHONY: version-bump
version-bump:
	@echo "$(YELLOW)$(BOLD)Bumping version$(RESET)"
	@if [ -f "$(VERSION_FILE)" ]; then \
		current_version=$$(cat $(VERSION_FILE)); \
		echo "Current version: $$current_version"; \
		echo "v4.0.2" > $(VERSION_FILE); \
		echo "New version: $$(cat $(VERSION_FILE))"; \
	else \
		echo "v4.0.2" > $(VERSION_FILE); \
		echo "Created version file with: $$(cat $(VERSION_FILE))"; \
	fi


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


.PHONY: build-and-push-optimized
build-and-push-optimized: version-bump docker-build docker-push
	@echo "$(GREEN)$(BOLD)Successfully built and pushed optimized image: $(FULL_IMAGE_TAG)$(RESET)"


.PHONY: docker-info
docker-info:
	@echo "Registry: $(REGISTRY)"
	@echo "Image Name: $(IMAGE_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Full Tag: $(FULL_IMAGE_TAG)"


# --------------------------------------------------------------------------------------
# Kubernetes/RayService deploy + test helpers
# --------------------------------------------------------------------------------------

# Ray/K8s variables (override at invocation if needed)
SERVICE_NAME ?= embedding-service
RAYSERVICE_FILE ?= k8s/rayservice-local.yaml
NAMESPACE ?= default

# Ingress testing vars
# If NODE_IP is not provided, we'll try to detect it at runtime
NODE_PORT ?= 30333
TEST_REQUESTS ?= 200
TEST_CONCURRENCY ?= 10
TEST_SEGMENTATION ?= false
USE_INGRESS_ONLY ?= true
INGRESS_WAIT_SECS ?= 60
INGRESS_POST_FALLBACK_WAIT_SECS ?= 90

.PHONY: _ensure-tools
_ensure-tools:
	@command -v kubectl >/dev/null 2>&1 || { echo "kubectl not found in PATH"; exit 1; }
	@command -v curl >/dev/null 2>&1 || { echo "curl not found in PATH"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found in PATH"; exit 1; }

# Ray dashboard port-forward settings
DASHBOARD_PORT ?= 8265
DASHBOARD_PID_FILE ?= run/dashboard_pf.pid

.PHONY: port-forward-dashboard
.SILENT: port-forward-dashboard
port-forward-dashboard: _ensure-tools
	@set -euo pipefail; \
	printf "%b\n" "$(YELLOW)$(BOLD)[dashboard] Starting port-forward to Ray Dashboard on port $(DASHBOARD_PORT)$(RESET)"; \
	mkdir -p run logs; \
	if [ -f "$(DASHBOARD_PID_FILE)" ]; then \
	  OLD_PID=$$(cat "$(DASHBOARD_PID_FILE)" 2>/dev/null || true); \
	  if [ -n "$$OLD_PID" ] && ps -p "$$OLD_PID" >/dev/null 2>&1; then \
	    printf "%b\n" "$(BLUE)$(BOLD)[dashboard] Existing port-forward detected (PID=$$OLD_PID). Leaving it running.$(RESET)"; \
	    exit 0; \
	  fi; \
	fi; \
	HEAD_RES=$$(kubectl get pods -n $(NAMESPACE) -l ray.io/node-type=head -o name | head -n1); \
	printf "%b\n" "$(BLUE)$(BOLD)[dashboard] Using resource: $$HEAD_RES$(RESET)"; \
	if [ -z "$$HEAD_RES" ]; then \
	  printf "%b\n" "$(RED)$(BOLD)[dashboard] Could not find Ray head pod in namespace $(NAMESPACE)$(RESET)"; \
	  exit 1; \
	fi; \
	LOG_FILE=logs/portforward-dashboard-$(DASHBOARD_PORT).log; \
	nohup kubectl -n $(NAMESPACE) port-forward "$$HEAD_RES" $(DASHBOARD_PORT):$(DASHBOARD_PORT) --address 0.0.0.0 > "$$LOG_FILE" 2>&1 & \
	PF_PID=$$!; \
	echo "$$PF_PID" > "$(DASHBOARD_PID_FILE)"; \
	sleep 1; \
	if ps -p "$$PF_PID" >/dev/null 2>&1; then \
	  printf "%b\n" "$(GREEN)$(BOLD)[dashboard] Port-forward running (PID=$$PF_PID). Logs: $$LOG_FILE$(RESET)"; \
	  printf "%b\n" "$(GREEN)$(BOLD)[dashboard] Access the dashboard at: http://127.0.0.1:$(DASHBOARD_PORT) or your host IP$(RESET)"; \
	else \
	  printf "%b\n" "$(RED)$(BOLD)[dashboard] Port-forward failed to start. See $$LOG_FILE$(RESET)"; \
	  exit 1; \
	fi

.PHONY: port-forward-dashboard-stop
.SILENT: port-forward-dashboard-stop
port-forward-dashboard-stop:
	@set -euo pipefail; \
	if [ ! -f "$(DASHBOARD_PID_FILE)" ]; then \
	  printf "%b\n" "$(BLUE)$(BOLD)[dashboard] No PID file found (nothing to stop)$(RESET)"; \
	  exit 0; \
	fi; \
	PID=$$(cat "$(DASHBOARD_PID_FILE)" 2>/dev/null || true); \
	if [ -z "$$PID" ]; then \
	  printf "%b\n" "$(BLUE)$(BOLD)[dashboard] PID file empty (nothing to stop)$(RESET)"; \
	  rm -f "$(DASHBOARD_PID_FILE)"; \
	  exit 0; \
	fi; \
	if ps -p "$$PID" >/dev/null 2>&1; then \
	  printf "%b\n" "$(YELLOW)$(BOLD)[dashboard] Stopping port-forward (PID=$$PID)$(RESET)"; \
	  kill "$$PID" || true; \
	  sleep 1; \
	  if ps -p "$$PID" >/dev/null 2>&1; then \
	    printf "%b\n" "$(YELLOW)$(BOLD)[dashboard] Force killing PID $$PID$(RESET)"; \
	    kill -9 "$$PID" || true; \
	  fi; \
	else \
	  printf "%b\n" "$(BLUE)$(BOLD)[dashboard] No running process for PID $$PID$(RESET)"; \
	fi; \
	rm -f "$(DASHBOARD_PID_FILE)"; \
	printf "%b\n" "$(GREEN)$(BOLD)[dashboard] Port-forward stopped$(RESET)"

.PHONY: _detect-node-ip
_detect-node-ip:
	@if [ -z "$(NODE_IP)" ]; then \
	  DETECTED_IP=$$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($$i=="src") {print $$(i+1); exit}}'); \
	  if [ -z "$$DETECTED_IP" ]; then \
	    echo "$(RED)$(BOLD)Unable to auto-detect NODE_IP; please pass NODE_IP=...$(RESET)"; \
	    exit 1; \
	  fi; \
	  echo "Using auto-detected NODE_IP=$$DETECTED_IP"; \
	  echo "NODE_IP=$$DETECTED_IP" > .make.cached.env; \
	else \
	  echo "NODE_IP=$(NODE_IP)" > .make.cached.env; \
	fi

.PHONY: redeploy-clean
redeploy-clean: _ensure-tools
	@set -euo pipefail; \
	printf "%b\n" "$(YELLOW)$(BOLD)[redeploy-clean] Deleting RayService $(SERVICE_NAME)$(RESET)"; \
	kubectl delete rayservice $(SERVICE_NAME) -n $(NAMESPACE) --ignore-not-found=true || true; \
	printf "%b\n" "$(YELLOW)$(BOLD)[redeploy-clean] Waiting briefly for pods to terminate$(RESET)"; \
	sleep 3; \
	PODS=$$(kubectl get pods -n $(NAMESPACE) -l ray.io/is-ray-node=yes -o name 2>/dev/null || true); \
		if [ -n "$$PODS" ]; then \
			printf "%b\n" "Found lingering Ray pods, forcing deletion:"; printf "%b\n" "$$PODS"; \
	  kubectl delete $$PODS -n $(NAMESPACE) --force --grace-period=0 || true; \
	fi; \
		printf "%b\n" "$(YELLOW)$(BOLD)[redeploy-clean] Verifying no Ray pods remain$(RESET)"; \
	kubectl get pods -n $(NAMESPACE) -l ray.io/is-ray-node=yes || true; \
	if command -v nvidia-smi >/dev/null 2>&1; then \
			printf "%b\n" "$(YELLOW)$(BOLD)[redeploy-clean] Checking GPU processes$(RESET)"; \
	  nvidia-smi || true; \
	fi

.PHONY: deploy-apply
deploy-apply: _ensure-tools
		@set -euo pipefail; \
		printf "%b\n" "$(YELLOW)$(BOLD)[deploy-apply] Applying $(RAYSERVICE_FILE)$(RESET)"; \
		kubectl apply -n $(NAMESPACE) -f $(RAYSERVICE_FILE); \
		printf "%b\n" "$(YELLOW)$(BOLD)[deploy-apply] Waiting for Ray pods to appear$(RESET)"; \
	for i in $$(seq 1 60); do \
	  CNT=$$(kubectl get pods -n $(NAMESPACE) -l ray.io/is-ray-node=yes -o name 2>/dev/null | wc -l | tr -d ' '); \
			if [ "$$CNT" -gt 0 ]; then printf "%b\n" "Found $$CNT Ray pods"; break; fi; \
	  sleep 2; \
	done; \
		printf "%b\n" "$(YELLOW)$(BOLD)[deploy-apply] Waiting for Ray pods Ready (timeout 10m)$(RESET)"; \
	kubectl wait --for=condition=ready pod -n $(NAMESPACE) -l ray.io/is-ray-node=yes --timeout=600s; \
		printf "%b\n" "$(YELLOW)$(BOLD)[deploy-apply] Waiting for Serve endpoints to be ready$(RESET)"; \
	for i in $$(seq 1 150); do \
	  EP=$$(kubectl get endpoints $(SERVICE_NAME)-serve-svc -n $(NAMESPACE) -o jsonpath='{.subsets[0].addresses[*].ip}' 2>/dev/null || true); \
			if [ -n "$$EP" ]; then printf "%b\n" "Serve endpoints: $$EP"; break; fi; \
	  sleep 2; \
	done; \
		printf "%b\n" "$(GREEN)$(BOLD)[deploy-apply] Ray pods Ready$(RESET)"; \
	kubectl get pods -n $(NAMESPACE) -o wide | sed -n '1,200p'

.PHONY: smoke-health
smoke-health: _ensure-tools _detect-node-ip
		@set -euo pipefail; \
		. ./.make.cached.env; \
		IP_DASH=$${NODE_IP//./-}; HOST=embeddings.$$IP_DASH.nip.io; \
		ENDPOINT_IP=http://$${NODE_IP}:$(NODE_PORT); \
		printf "%b\n" "$(YELLOW)$(BOLD)[smoke-health] Checking health via Ingress: $$ENDPOINT_IP/health (Host: $$HOST)$(RESET)"; \
		RESOLVE=$$(getent hosts $$HOST 2>/dev/null | awk '{print $$1}' | head -n1 || true); \
		if [ -n "$$RESOLVE" ] && [ "$$RESOLVE" != "$$NODE_IP" ]; then \
			printf "%b\n" "$(YELLOW)$(BOLD)[smoke-health] Warning: $$HOST resolves to $$RESOLVE, expected $$NODE_IP$(RESET)"; \
		fi; \
		CODE=$$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $$HOST" $$ENDPOINT_IP/health || true); \
		RES=$$(curl -s -H "Host: $$HOST" $$ENDPOINT_IP/health || true); \
		printf "%b\n" "Health code: $$CODE body: $$RES"; \
		if [ "$$CODE" != "200" ] || ! echo "$$RES" | grep -qi '"status"\s*:\s*"healthy"'; then \
			printf "%b\n" "$(RED)$(BOLD)[smoke-health] Ingress not healthy (code=$$CODE), trying direct pod endpoint fallback$(RESET)"; \
			POD_IP=$$(kubectl get endpoints embedding-service-serve-svc -n $(NAMESPACE) -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true); \
			if [ -n "$$POD_IP" ]; then \
				printf "%b\n" "Trying http://$$POD_IP:8000/health"; \
				CODE=$$(curl -s -o /dev/null -w "%{http_code}" http://$$POD_IP:8000/health || true); \
				RES=$$(curl -s http://$$POD_IP:8000/health || true); \
				printf "%b\n" "Fallback code: $$CODE body: $$RES"; \
				if [ "$$CODE" != "200" ] || ! echo "$$RES" | grep -qi '"status"\s*:\s*"healthy"'; then \
					printf "%b\n" "$(RED)$(BOLD)[smoke-health] Fallback health also not healthy$(RESET)"; \
					exit 1; \
				fi; \
				printf "%b\n" "$(BLUE)$(BOLD)[smoke-health] Fallback OK; waiting up to $(INGRESS_POST_FALLBACK_WAIT_SECS)s for Ingress to become healthy$(RESET)"; \
				START=$$(date +%s); OK=false; \
				while :; do \
				  CODE=$$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $$HOST" $$ENDPOINT_IP/health || true); \
				  RES=$$(curl -s -H "Host: $$HOST" $$ENDPOINT_IP/health || true); \
				  if [ "$$CODE" = "200" ] && echo "$$RES" | grep -qi '"status"\s*:\s*"healthy"'; then OK=true; break; fi; \
				  NOW=$$(date +%s); ELAPSED=$$((NOW-START)); \
				  if [ "$$ELAPSED" -ge "$(INGRESS_POST_FALLBACK_WAIT_SECS)" ]; then break; fi; \
				  sleep 2; \
				done; \
				if [ "$$OK" != "true" ]; then \
				  printf "%b\n" "$(RED)$(BOLD)[smoke-health] Ingress still not healthy after $(INGRESS_POST_FALLBACK_WAIT_SECS)s (last code=$$CODE). Aborting.$(RESET)"; \
				  exit 1; \
				fi; \
			else \
				printf "%b\n" "$(RED)$(BOLD)[smoke-health] Could not determine pod IP for fallback$(RESET)"; \
				exit 1; \
			fi; \
		fi; \
		printf "%b\n" "$(GREEN)$(BOLD)[smoke-health] Health OK$(RESET)"

.PHONY: serve-status
.SILENT: serve-status
serve-status:
	@set -euo pipefail; \
	HEAD_POD=$$(kubectl get pods -n $(NAMESPACE) -l ray.io/node-type=head -o name | head -n1 | cut -d/ -f2); \
	if [ -z "$$HEAD_POD" ]; then echo "No Ray head pod found"; exit 1; fi; \
	printf "%b\n" "$(YELLOW)$(BOLD)[serve-status] Querying Serve status from $$HEAD_POD$(RESET)"; \
	kubectl exec -n $(NAMESPACE) $$HEAD_POD -- python3 -c 'import json, dataclasses as dc; from ray import serve; s=serve.status(); d = s.to_dict() if hasattr(s, "to_dict") else dc.asdict(s); print(json.dumps(d, indent=2))'

.PHONY: quick-load-test
quick-load-test: _ensure-tools _detect-node-ip
		@set -euo pipefail; \
		. ./.make.cached.env; \
		IP_DASH=$${NODE_IP//./-}; HOST=embeddings.$$IP_DASH.nip.io; \
	TARGET_URL=http://$$HOST:$(NODE_PORT); \
	HEALTH_URL=http://$${NODE_IP}:$(NODE_PORT); \
	printf "%b\n" "$(BLUE)$(BOLD)[quick-load-test] Debug: HOST=$$HOST TARGET_URL=$$TARGET_URL HEALTH_URL=$$HEALTH_URL$(RESET)"; \
	# Prefer ingress: health wait (skip if SKIP_HEALTH_WAIT=true)
	if [ "$${SKIP_HEALTH_WAIT:-false}" != "true" ]; then \
	  START=$$(date +%s); \
	  OK=false; \
	  while :; do \
	    CODE=$$(curl -s -o /dev/null -w "%{http_code}" -H "Host: $$HOST" $$HEALTH_URL/health || true); \
	    RES=$$(curl -s -H "Host: $$HOST" $$HEALTH_URL/health || true); \
	    if [ "$$CODE" = "200" ] && echo "$$RES" | grep -qi '"status"\s*:\s*"healthy"'; then OK=true; break; fi; \
	    NOW=$$(date +%s); \
	    ELAPSED=$$((NOW-START)); \
	    if [ "$$ELAPSED" -ge "$(INGRESS_WAIT_SECS)" ]; then break; fi; \
	    sleep 2; \
	  done; \
	  if [ "$$OK" != "true" ]; then \
	    if [ "$(USE_INGRESS_ONLY)" = "true" ]; then \
	      printf "%b\n" "$(RED)$(BOLD)[quick-load-test] Ingress not healthy after $(INGRESS_WAIT_SECS)s (last code=$$CODE). Aborting per USE_INGRESS_ONLY=true$(RESET)"; \
	      exit 1; \
	    else \
	      POD_IP=$$(kubectl get endpoints $(SERVICE_NAME)-serve-svc -n $(NAMESPACE) -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true); \
	      if [ -n "$$POD_IP" ]; then \
	        TARGET_URL=http://$$POD_IP:8000; \
	        printf "%b\n" "Ingress not ready (last code=$$CODE), falling back to direct pod URL: $$TARGET_URL"; \
	      fi; \
	    fi; \
	  fi; \
	fi; \
	printf "%b\n" "$(YELLOW)$(BOLD)[quick-load-test] Running load test to $$TARGET_URL: R=$(TEST_REQUESTS) C=$(TEST_CONCURRENCY) segmentation=$(TEST_SEGMENTATION)$(RESET)"; \
	python3 working_dir/load_test.py --url $$TARGET_URL --requests $(TEST_REQUESTS) --concurrency $(TEST_CONCURRENCY) --segmentation $(TEST_SEGMENTATION)

.PHONY: quick-load-test-ingress
quick-load-test-ingress: smoke-health
	@set -euo pipefail; \
	. ./.make.cached.env; \
	IP_DASH=$${NODE_IP//./-}; HOST=embeddings.$$IP_DASH.nip.io; \
	URL=http://$$HOST:$(NODE_PORT); \
	printf "%b\n" "$(YELLOW)$(BOLD)[quick-load-test-ingress] Running ingress-only load test: $$URL R=$(TEST_REQUESTS) C=$(TEST_CONCURRENCY) segmentation=$(TEST_SEGMENTATION)$(RESET)"; \
	python3 working_dir/load_test.py --url $$URL --requests $(TEST_REQUESTS) --concurrency $(TEST_CONCURRENCY) --segmentation $(TEST_SEGMENTATION)

.PHONY: redeploy
redeploy: redeploy-clean deploy-apply smoke-health
	@printf "%b\n" "$(GREEN)$(BOLD)[redeploy] Completed cleanup, deploy, and health verification$(RESET)"

.PHONY: redeploy-and-test
redeploy-and-test: redeploy quick-load-test-ingress
	@printf "%b\n" "$(GREEN)$(BOLD)[redeploy-and-test] Completed end-to-end redeploy with ingress load test$(RESET)"

.PHONY: redeploy-and-test-ingress
redeploy-and-test-ingress: redeploy quick-load-test-ingress
	@printf "%b\n" "$(GREEN)$(BOLD)[redeploy-and-test-ingress] Completed end-to-end redeploy with ingress load test$(RESET)"

.PHONY: redeploy-and-test-fallback
redeploy-and-test-fallback: redeploy
	@$(MAKE) quick-load-test USE_INGRESS_ONLY=false SKIP_HEALTH_WAIT=true
	@printf "%b\n" "$(GREEN)$(BOLD)[redeploy-and-test-fallback] Completed end-to-end redeploy with fallback load test$(RESET)"
