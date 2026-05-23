# ── Base image ─────────────────────────────────────────────────────────────────
# CUDA 11.8 + cuDNN 8 on Ubuntu 22.04 — required for bitsandbytes 4-bit on HF Spaces T4
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── System deps ────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# ── HF Spaces requires a non-root user ─────────────────────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# ── Python deps ────────────────────────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY --chown=user . .

# ── HF Spaces port ─────────────────────────────────────────────────────────────
# HF Spaces expects port 7860 — NOT 8000
EXPOSE 7860

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# start.sh pulls models from HF Hub then launches uvicorn
CMD ["bash", "start.sh"]