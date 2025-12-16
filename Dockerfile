FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Define Hugging Face Cache Directory (Fixed location for pre-downloading)
ENV HF_HOME="/root/.cache/huggingface"

# Set the working directory
WORKDIR /workspace

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    pandas \
    numpy \
    decord \
    tqdm \
    scikit-learn \
    xgboost \
    opencv-python-headless \
    sentencepiece \
    protobuf \
    timm \
    einops \
    huggingface_hub

RUN python3 -c "from huggingface_hub import snapshot_download; \
    print('Downloading OpenGVLab/VideoMAEv2-Base...'); \
    snapshot_download(repo_id='OpenGVLab/VideoMAEv2-Base', \
                      repo_type='model', \
                      ignore_patterns=['*.msgpack', '*.h5', '*.ot'])"

# Set Offline Mode
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY scripts/train_phase2_vision.py /workspace/train_phase2_ddp.py
COPY data /workspace/data
COPY models /workspace/models
COPY scripts /workspace/scripts

ENTRYPOINT ["torchrun", "--nproc_per_node=4", "train_phase2_ddp.py"]