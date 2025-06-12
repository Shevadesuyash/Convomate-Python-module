# Stage 1: Build and Cache Stage
# We use a specific version of Python for reproducibility.
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Install build-time and runtime OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc default-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements and the new caching script
COPY requirements.txt cache_models.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the caching script to download all models and dependencies.
# This populates the cache directories.
RUN python cache_models.py

# ---

# Stage 2: Runtime Stage
# Start from a clean, lightweight Python image.
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install only the necessary runtime OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 default-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment with all packages from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the pre-downloaded model caches from the builder stage.
# This is the key step to prevent runtime downloads.
COPY --from=builder /root/.cache /root/.cache

# Set the PATH to include the virtual environment's binaries.
ENV PATH="/opt/venv/bin:$PATH"

# Set Hugging Face cache environment variable to the copied location
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Copy the rest of the application code into the container.
COPY . .

# Set environment variables for production.
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose the port the application will run on.
EXPOSE 8080

# Healthcheck to ensure the application is running correctly.
HEALTHCHECK --interval=60s --timeout=15s --start-period=45s \
  CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application using Gunicorn.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]