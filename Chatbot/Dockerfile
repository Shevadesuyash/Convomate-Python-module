# Start from a lightweight base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install basic system dependencies (needed for torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy only needed source files
COPY app.py .
COPY grammar_chatbot.py .

# Expose the port your Flask app runs on
EXPOSE 5003

# Run the Flask app
CMD ["python", "app.py"]
