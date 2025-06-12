# Stage 1: Builder stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies with OpenJDK 17 (available in Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user -r requirements.txt

# Cache models during build
COPY cache_models.py .
RUN python cache_models.py

# Stage 2: Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create a non-root user
RUN useradd -m myuser && chown -R myuser /app
USER myuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]