# --- Stage 1: Build & Download Models ---
FROM python:3.10-slim AS builder

# Install system dependencies needed for both Python packages and LanguageTool
# openjdk-17-jre-headless for LanguageTool's Java dependency
# build-essential and libglib2.0-0, etc., for potential Python package compilation (e.g., torch)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for the builder stage
WORKDIR /app

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy paragraph_checker.py to the builder stage
COPY paragraph_checker.py .

# Pre-download LanguageTool dictionaries and ML models during build
# This makes them part of the image, reducing startup time, but increasing image size.
# The multi-stage build will help mitigate the final image size.
RUN python -c "from paragraph_checker import initialize_models; initialize_models()"

# --- Stage 2: Final Runtime Image ---
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory for the final image
WORKDIR /app

# Only copy runtime system dependencies (often fewer than build dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary Python site-packages from the builder stage
# This includes all installed Python libraries and the downloaded ML models.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /root/.cache/language_tool_python /root/.cache/language_tool_python # LanguageTool cache
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface             # HuggingFace model cache

# Copy application code from your project
COPY app.py .
COPY paragraph_checker.py . # This will include the logic for initialize_models and model usage

# Expose the port your Flask app runs on
EXPOSE 5001

# Command to run the Flask application
# Use Gunicorn for production for better performance and robustness
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
# For local development or simple Free Tier EC2, your current CMD is fine:
CMD ["python", "app.py"]