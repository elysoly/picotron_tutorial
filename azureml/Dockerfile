# Base already ships torch 2.1.0 + CUDA 12.1 + Python 3.10, matching requirements.txt.
# The -devel variant is required: flash-attn's triton kernels need nvcc at runtime.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Everything from requirements.txt except torch (in the base) and flash-attn (wheel below).
RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        datasets==2.19.1 \
        transformers==4.41.1 \
        lovely_tensors \
        wandb

# Prebuilt flash-attn wheel. Building 2.5.0 from source takes ~1h and >32GB RAM;
# this wheel matches the base exactly: cu12x / torch2.1 / cxx11abiFALSE / cp310.
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.0/flash_attn-2.5.0+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Fail the image build rather than the job if the wheel is ABI-mismatched.
RUN python -c "import torch, flash_attn; from flash_attn.flash_attn_interface import flash_attn_func; from flash_attn.ops.triton.layer_norm import layer_norm_fn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"

WORKDIR /workspace
