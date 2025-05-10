# Use an official Python runtime as a parent image
FROM python:3.12

WORKDIR /app

# Install system dependencies if needed (e.g., git)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install diffsynth from source
RUN git clone https://github.com/modelscope/DiffSynth-Studio.git \
    && cd DiffSynth-Studio \
    && pip install -e . \
    && cd /app

# Download video models using modelscope
# Ensure the models directory exists and is writable if models are large
RUN mkdir -p models/Wan-AI/Wan2.1-T2V-1.3B
RUN python -c "from modelscope import snapshot_download; snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', local_dir='models/Wan-AI/Wan2.1-T2V-1.3B')"

# Copy the rest of the application code
COPY video_api.py ./video_api.py
COPY saved_videos ./saved_videos
COPY safe_tensors ./safe_tensors

# Expose the port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "video_api:app", "--host", "0.0.0.0", "--port", "8000"]

