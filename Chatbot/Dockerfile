# Stage 1: Builder - installs dependencies and pre-caches models
FROM python:3.9-slim-buster AS builder

# Set working directory
WORKDIR /app

# Install system dependencies needed by language_tool_python, transformers, torch etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements initially for better cache leverage
COPY Paragraph-Checker-And-Chatbot/requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application source code
COPY Paragraph-Checker-And-Chatbot/ /app/

# Pre-cache models by running the cache_models script (adapted from pre_cache_models function)
# Create a small script to cache models at build time
RUN python -c "\
import language_tool_python; \
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
models = ['vennify/t5-base-grammar-correction','humarin/chatgpt_paraphraser_on_T5_base']; \
print('Caching LanguageTool model...'); \
language_tool_python.LanguageTool('en-US'); \
print('Caching transformer models...'); \
for m in models: \
    AutoTokenizer.from_pretrained(m); \
    AutoModelForSeq2SeqLM.from_pretrained(m); \
print('Model caching completed.')"

# Stage 2: Final image - minimal runtime environment
FROM python:3.9-slim-buster

WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source code
COPY --from=builder /app /app

# Expose port 8080 for Flask app
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the Flask application
CMD ["python", "app.py"]