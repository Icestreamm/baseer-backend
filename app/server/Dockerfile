FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Copy YOLO model file
COPY handle_best.pt .

# Expose port (Render will auto-assign, but 9001 is default)
EXPOSE 9001

# Run the application
CMD ["python", "app.py"]
