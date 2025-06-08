FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set up application directory
WORKDIR /app
COPY . .

# Writable cache directory for Hugging Face & LanguageTool
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV LANGTOOL_HOME=/app/cache/langtool

RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Do NOT pre-download models during build to avoid /.cache issues
# RUN python -c "from paragraph_checker import initialize_models; initialize_models()"

EXPOSE 5001
CMD ["python", "app.py"]
