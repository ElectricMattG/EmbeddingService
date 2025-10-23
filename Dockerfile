# Simple Python container for EmbeddingService
# Uses host CUDA via nvidia-container-toolkit
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install wget for Ray health probes
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY version .
COPY setup.py .

# Install the application
RUN pip install -e .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8090

# Set GPU device (can be overridden at runtime)
ENV CUDA_VISIBLE_DEVICES=0

# Run the service
CMD ["python", "-m", "uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8090"]
