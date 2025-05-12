# Base image with CUDA 11.8 and Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and curl
RUN apt-get update && \
   apt-get install -y --no-install-recommends \
   git \
   unzip \
   curl \
   bzip2 \
   ca-certificates \
   libglib2.0-0 \
   libxext6 \
   libsm6 \
   libxrender1 && \
   rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN $CONDA_DIR/bin/conda init bash

# Create conda environment
RUN conda create -y -n diffusion-pipe python=3.12 && conda clean -afy

# Activate the environment by default
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=diffusion-pipe
ENV PATH="$CONDA_DIR/envs/diffusion-pipe/bin:$PATH"
RUN echo "conda activate diffusion-pipe" >> ~/.bashrc

# Set base working directory
WORKDIR /app

# Copy only the trainig_pipeline folder from the local AdFame repository
# Assumes Dockerfile is in the AdFame directory, and trainig_pipeline is a subdirectory
COPY trainig_pipeline /app/AdFame/trainig_pipeline

# Set working directory for subsequent operations within the copied pipeline
WORKDIR /app/AdFame/trainig_pipeline

RUN cd diffusion-pipe


# RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


# Set working directory to diffusion-pipe (relative to current WORKDIR)
# Current WORKDIR is /app/AdFame/trainig_pipeline, so this becomes /app/AdFame/trainig_pipeline/diffusion-pipe


WORKDIR diffusion-pipe
# Since submodules are copied locally (as part of trainig_pipeline), git submodule commands are likely not needed here.

RUN pwd

# Install requirements for diffusion-pipe
RUN pip install -r requirements.txt

RUN python -m pip -v install flash-attn --no-build-isolation
# RUN pip install flash-attn --no-build-isolation
RUN pip install huggingface_hub[cli]
RUN pip install gdown
RUN pip install tensorboard
RUN pip install accelerate
# ADDED: Install Ray, MLflow, and Prometheus client
RUN pip install "ray[default]==2.42.1"
RUN pip install mlflow
RUN pip install prometheus_client
RUN pip install modelscope



# Change back to training_pipeline directory
# Current WORKDIR is /app/AdFame/trainig_pipeline/diffusion-pipe, so cd .. goes to /app/AdFame/trainig_pipeline
RUN pwd 

# Model download steps are removed as per user request.
# User will download the model manually inside the container.
# The application will expect the model at /app/AdFame/trainig_pipeline/diffusion-pipe/models/Wan2.1-T2V-14B
# Ensure the target directory for the model exists if other parts of the application expect it.
RUN mkdir -p /app/AdFame/trainig_pipeline/diffusion-pipe/models

RUN python -c "from modelscope import snapshot_download; snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', local_dir='models/Wan2.1-T2V-14B')"

# Echo "DONE" as in the script
RUN echo "DONE"

# Suggest a CMD if the user wants to run something by default, e.g.
# For your current need (manual installations first), this CMD should be overridden in docker-compose.yml
# CMD ["bash"]

CMD ["tail", "-f", "/dev/null"]
