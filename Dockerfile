# Base image with CUDA 11.8 and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04


# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive


# Install system dependencies, Python 3.8 from PPA, and curl
RUN apt-get update && \
   apt-get install -y --no-install-recommends \
   git \
   unzip \
   curl \
   software-properties-common && \
   add-apt-repository -y ppa:deadsnakes/ppa && \
   apt-get update && \
   apt-get install -y --no-install-recommends \
   python3.8 \
   python3.8-dev \
   python3.8-venv && \
   rm -rf /var/lib/apt/lists/*


# Install pip for Python 3.8 using the correct script and upgrade it
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 && \
   python3.8 -m pip install --upgrade pip


# Verify Python and pip versions (optional, good for debugging)
RUN python3.8 --version && python3.8 -m pip --version


# Set base working directory
WORKDIR /app


# Copy only the trainig_pipeline folder from the local AdFame repository
# Assumes Dockerfile is in the AdFame directory, and trainig_pipeline is a subdirectory
COPY trainig_pipeline /app/AdFame/trainig_pipeline


# Set working directory for subsequent operations within the copied pipeline
WORKDIR /app/AdFame/trainig_pipeline


# Create Python virtual environment using Python 3.8
RUN python3.8 -m venv venv


# Add venv to PATH for subsequent RUN commands
ENV PATH="/app/AdFame/trainig_pipeline/venv/bin:$PATH"


# Upgrade pip within the venv
RUN pip install --upgrade pip


# Install Python packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install huggingface_hub[cli]
RUN pip install gdown
RUN pip install tensorboard
RUN pip install accelerate
# Flash attention installation is intentionally ignored as per user request:
# RUN pip install flash-attn --no-build-isolation


# Set working directory to diffusion-pipe (relative to current WORKDIR)
# Current WORKDIR is /app/AdFame/trainig_pipeline, so this becomes /app/AdFame/trainig_pipeline/diffusion-pipe
WORKDIR diffusion-pipe
# Since submodules are copied locally (as part of trainig_pipeline), git submodule commands are likely not needed here.


# Install requirements for diffusion-pipe
RUN pip install -r requirements.txt


# Change back to training_pipeline directory
# Current WORKDIR is /app/AdFame/trainig_pipeline/diffusion-pipe, so cd .. goes to /app/AdFame/trainig_pipeline
WORKDIR ..


# Model download steps are removed as per user request.
# User will download the model manually inside the container.
# The application will expect the model at /app/AdFame/trainig_pipeline/diffusion-pipe/models/Wan2.1-T2V-14B
# Ensure the target directory for the model exists if other parts of the application expect it.
RUN mkdir -p /app/AdFame/trainig_pipeline/diffusion-pipe/models


# Echo "DONE" as in the script
RUN echo "DONE"


# Suggest a CMD if the user wants to run something by default, e.g.
# CMD ["bash"]





