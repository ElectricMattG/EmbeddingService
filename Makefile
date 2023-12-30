VIRTUALENV_NAME := $(shell basename $(CURDIR))

PIP_COMPILE_ARGS = -v -U --no-emit-index-url --no-emit-trusted-host --strip-extras --resolver=backtracking

PYTHON_VERSION = 3.9.9 # keep minor version in sync with tfx docker image (see Dockerfile for TFX release on GitHub)

.PHONY: _python_version
_python_version:
	@if [ -z $(PYTHON_VERSION) ]; then \
		echo "$(RED)$(BOLD)PYTHON_VERSION should be set when including python make rules$(RESET)"; \
		exit 1; \
	fi


.PHONY: pyenv
pyenv: | _python_version # Create the pyenv virtual environment.
	@echo "$(YELLOW)$(BOLD)Creating virtual environment$(RESET)"
	pyenv install -s $(PYTHON_VERSION)
	pyenv virtualenv $(PYTHON_VERSION) $(VIRTUALENV_NAME)
	pyenv local $(VIRTUALENV_NAME)
	pip install --upgrade pip


.PHONY: pre-commit
pre-commit:
	@pip install -q pre-commit
	@pre-commit install


.PHONY: check
check: | pre-commit # Run checks (pre-commit).
	@echo "$(YELLOW)$(BOLD)Running pre-commit checks$(RESET)"
	pre-commit run --all-files


.PHONY: pip-tools
pip-tools:
	@pip install -q pip-tools


%.txt: %.in | pip-tools # Compile specified requirements (pip-compile).
	@echo "$(YELLOW)$(BOLD)Compiling $*.txt$(RESET)"
	CUSTOM_COMPILE_COMMAND="make update" pip-compile $(PIP_COMPILE_ARGS) $*.in


.PHONY: requirements
requirements: requirements.in dev-requirements.in | pip-tools # Compile all requirements (pip-compile).
	@echo "$(YELLOW)$(BOLD)Compiling requirements$(RESET)"
	CUSTOM_COMPILE_COMMAND="make update" pip-compile $(PIP_COMPILE_ARGS) requirements.in --output-file requirements.txt
	CUSTOM_COMPILE_COMMAND="make update" pip-compile $(PIP_COMPILE_ARGS) dev-requirements.in --output-file dev-requirements.txt


.PHONY: sync
sync: requirements.txt dev-requirements.txt | pip-tools # Sync dependencies (pip-sync).
	@echo "$(YELLOW)$(BOLD)Syncing dependencies$(RESET)"
	pip-sync requirements.txt dev-requirements.txt


.PHONY: install
install: requirements.txt dev-requirements.txt # Install dependencies (pip).
	@echo "$(YELLOW)$(BOLD)Installing dependencies$(RESET)"
	pip install \
	-r requirements.txt -r dev-requirements.txt
	pip install -e .


.PHONY: test
test: | install # Run tests (pytest).
	@echo "$(YELLOW)$(BOLD)Running tests$(RESET)"
	pytest
