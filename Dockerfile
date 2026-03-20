# ─────────────────────────────────────────────────────
# Telegram Vision Bot — Dockerfile
# Multi-stage build: CPU-only (lightweight) by default
# ─────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System deps for Pillow and Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY app.py .
COPY .env.example .env.example

# Model cache directory (mount as volume in production)
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Expose Gradio port
EXPOSE 7860

# Default: bot only
CMD ["python", "app.py"]