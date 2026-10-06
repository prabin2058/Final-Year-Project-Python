FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<70.0.0" wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose Flask port
EXPOSE 5000

# Use entrypoint script to run DB setup then start Flask
CMD ["sh", "entrypoint.sh"]
