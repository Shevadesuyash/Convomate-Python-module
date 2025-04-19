# Use an official Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download necessary nltk data
RUN python -m nltk.downloader punkt

# Copy app files
COPY . .

# Expose the port
EXPOSE 5002

# Run the app
CMD ["python", "app.py"]
