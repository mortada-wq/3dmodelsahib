# RunPod Serverless worker — TRELLIS.2 image-to-3D
#
# This repo is the deployment source: RunPod builds it from the repo root,
# with Dockerfile path /Dockerfile. The TRELLIS.2 source lives in this repo,
# so the image installs it from ./trellis2 and ./o-voxel instead of cloning.
#
# Build must be linux/amd64. The CUDA extensions compile for the
# architectures in TORCH_CUDA_ARCH_LIST.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRELLIS_PATH=/app/TRELLIS.2 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel ninja packaging

# TRELLIS.2 targets torch 2.6.0/cu124; the RunPod base image ships 2.4.0.
RUN pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
        runpod \
        requests \
        pillow \
        numpy \
        opencv-python-headless \
        imageio \
        imageio-ffmpeg \
        trimesh \
        transformers \
        accelerate \
        safetensors \
        huggingface_hub \
        hf_transfer \
        easydict \
        einops \
        kornia \
        timm \
        omegaconf \
        tqdm \
        zstandard \
        plyfile \
        pymeshlab \
        open3d \
        lpips \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# flash-attn is the slow compile (20-40 min). Use --build-arg
# ATTN_BACKEND=xformers on GPUs that cannot run it.
ARG ATTN_BACKEND=flash
RUN if [ "$ATTN_BACKEND" = "flash" ]; then \
        pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation; \
    else \
        pip install --no-cache-dir "xformers==0.0.29.post3"; \
    fi

# o_voxel/postprocess.py imports nvdiffrast at module level, so GLB export
# needs it even though this worker never renders a preview.
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec \
    && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrec \
    && rm -rf /tmp/nvdiffrec

RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast \
    && rm -rf /tmp/nvdiffrast

RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/CuMesh \
    && rm -rf /tmp/CuMesh

RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/FlexGEMM \
    && rm -rf /tmp/FlexGEMM

# The repo carries TRELLIS.2 source and handler.py at the root. .git is
# excluded from the build context by .dockerignore.
COPY . /app/TRELLIS.2

# o-voxel's Eigen submodule is not included in a GitHub archive or in
# RunPod's clone, so fetch the pinned commit explicitly before building it.
RUN git init /tmp/eigen \
    && cd /tmp/eigen \
    && git remote add origin https://gitlab.com/libeigen/eigen.git \
    && git fetch --depth 1 origin 21e4582d1739107337a03460c81412981130373e \
    && git checkout FETCH_HEAD \
    && rm -rf /tmp/eigen/.git \
    && rm -rf /app/TRELLIS.2/o-voxel/third_party/eigen \
    && mv /tmp/eigen /app/TRELLIS.2/o-voxel/third_party/eigen

RUN pip install --no-cache-dir --no-build-isolation /app/TRELLIS.2/o-voxel

WORKDIR /app/TRELLIS.2
ENV PYTHONPATH=/app/TRELLIS.2
ENV ATTN_BACKEND=${ATTN_BACKEND}

CMD ["python", "-u", "/app/TRELLIS.2/handler.py"]
