# Stage 1: Builder stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jre-headless \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Pre-download and install LanguageTool
RUN mkdir -p /app/languagetool && \
    wget https://languagetool.org/download/LanguageTool-stable.zip -O /tmp/LanguageTool.zip && \
    unzip /tmp/LanguageTool.zip -d /app/languagetool && \
    rm /tmp/LanguageTool.zip

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy LanguageTool installation
COPY --from=builder /app/languagetool /app/languagetool

# Copy application code
COPY . .

# Set up cache directories
RUN mkdir -p /app/cache && \
    chmod -R 777 /app/cache
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV LANGUAGE_TOOL_HOME=/app/languagetool/LanguageTool-5.7

# Create a non-root user
RUN useradd -m myuser && \
    chown -R myuser:myuser /app && \
    chmod -R 755 /app
USER myuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["python", "app.py"]