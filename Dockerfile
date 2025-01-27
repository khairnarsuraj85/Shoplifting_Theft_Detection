# Use a lightweight Python base image
FROM python:3.9-slim

# Set environment variable to ensure non-interactive apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Copy application files to the working directory
COPY . /app

# Install necessary system dependencies for OpenCV and general functionality
RUN rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    --fix-missing && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the working directory has all the necessary model files (optional: pre-download or copy)
COPY best.pt /app/
COPY model_weights.json /app/

# Expose the Flask app's port
EXPOSE 5000

# Set the command to run the Flask application
CMD ["python", "app.py"]
