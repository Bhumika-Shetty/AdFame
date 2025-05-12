# Dockerfile for Triton Inference Server with DiffSynth Text-to-Video Model

# Use an official NVIDIA Triton Inference Server image as a parent image
# This version uses Python 3.8. Check compatibility with DiffSynth and ModelScope.
FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    # Add any other system dependencies required by modelscope or diffsynth
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements for the Python backend model
COPY ./requirements.txt /workspace/requirements.txt

# Install Python dependencies
# Ensure these are compatible with Python 3.8 from the base image
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Install DiffSynth-Studio from source
RUN git clone https://github.com/modelscope/DiffSynth-Studio.git /workspace/DiffSynth-Studio \
    && cd /workspace/DiffSynth-Studio \
    && pip install --no-cache-dir -e . \
    && cd /workspace

# Create directory for downloading models
ENV MODELS_DOWNLOAD_DIR=/models_download_cache
RUN mkdir -p ${MODELS_DOWNLOAD_DIR}/Wan-AI/Wan2.1-T2V-1.3B
RUN mkdir -p ${MODELS_DOWNLOAD_DIR}/safe_tensors

# Download main DiffSynth models using ModelScope
# Note: This command will run during the Docker build process.
# Ensure network connectivity and sufficient disk space during build.
RUN python -m modelscope.cli.download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir ${MODELS_DOWNLOAD_DIR}/Wan-AI/Wan2.1-T2V-1.3B

# --- Model Repository Setup ---
# Triton expects models in /models (or path specified by --model-repository)
ENV TRITON_MODEL_REPO_PATH=/opt/tritonserver/models
ENV MODEL_NAME=text_to_video_diffsynth
ENV MODEL_VERSION=1

# Create the target structure for the model files within the model repository
RUN mkdir -p ${TRITON_MODEL_REPO_PATH}/${MODEL_NAME}/${MODEL_VERSION}/model_files/Wan-AI/Wan2.1-T2V-1.3B
RUN mkdir -p ${TRITON_MODEL_REPO_PATH}/${MODEL_NAME}/${MODEL_VERSION}/model_files/safe_tensors

# Copy downloaded main models to the Triton model repository structure
RUN cp -r ${MODELS_DOWNLOAD_DIR}/Wan-AI/Wan2.1-T2V-1.3B/* ${TRITON_MODEL_REPO_PATH}/${MODEL_NAME}/${MODEL_VERSION}/model_files/Wan-AI/Wan2.1-T2V-1.3B/

# Copy LoRA model. Assumes adapter_model.safetensors is in a 'lora_files/safe_tensors' directory in the build context.
# This path needs to match where the user provides the LoRA file during build.
COPY ./lora_files/safe_tensors/adapter_model.safetensors ${TRITON_MODEL_REPO_PATH}/${MODEL_NAME}/${MODEL_VERSION}/model_files/safe_tensors/adapter_model.safetensors

# Copy the Python backend script (model.py) and model configuration (config.pbtxt)
# These files should be created based on previous steps and placed in the build context under
# e.g., context_root/triton_model_config/text_to_video_diffsynth/config.pbtxt
# and context_root/triton_model_config/text_to_video_diffsynth/1/model.py
COPY ./model_config/${MODEL_NAME}/config.pbtxt ${TRITON_MODEL_REPO_PATH}/${MODEL_NAME}/config.pbtxt
COPY ./model_config/${MODEL_NAME}/${MODEL_VERSION}/model.py ${TRITON_MODEL_REPO_PATH}/${MODEL_NAME}/${MODEL_VERSION}/model.py

# Expose Triton ports
EXPOSE 8000 # HTTP
EXPOSE 8001 # gRPC
EXPOSE 8002 # Metrics

# Set the default command to start Triton Inference Server
# Using --strict-model-config=false can be helpful for debugging initial setups.
# --log-verbose=1 for more detailed logs.
CMD ["tritonserver", "--model-repository=${TRITON_MODEL_REPO_PATH}", "--strict-model-config=false", "--log-verbose=1"] 

