#!/bin/bash
# MedQA-Hindi — HF Spaces Entrypoint
# Pulls models from HF Hub then starts FastAPI on port 7860

set -e  # exit on any error

echo "========================================"
echo "🏥 MedQA-Hindi Starting..."
echo "========================================"

# ── HF Hub model IDs ───────────────────────────────────────────────────────────
# Set these to your actual HF Hub repo IDs after uploading
QLORA_MODEL_ID="${QLORA_MODEL_ID:-vaibhavrakshe161/medqa-hindi-qlora}"
DPO_MODEL_ID="${DPO_MODEL_ID:-vaibhavrakshe161/medqa-hindi-dpo}"

QLORA_PATH="./models/qlora_merged"
DPO_PATH="./models/dpo_merged"

# ── Download models from HF Hub ────────────────────────────────────────────────
echo ""
echo "📥 Downloading models from HF Hub..."

# Download QLoRA model if not already cached
if [ ! -d "$QLORA_PATH" ]; then
    echo "   Downloading QLoRA model: $QLORA_MODEL_ID"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$QLORA_MODEL_ID',
    local_dir='$QLORA_PATH',
    local_dir_use_symlinks=False
)
print('   ✅ QLoRA model downloaded')
"
else
    echo "   ✅ QLoRA model already cached"
fi

# Download DPO model if not already cached
if [ ! -d "$DPO_PATH" ]; then
    echo "   Downloading DPO model: $DPO_MODEL_ID"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$DPO_MODEL_ID',
    local_dir='$DPO_PATH',
    local_dir_use_symlinks=False
)
print('   ✅ DPO model downloaded')
"
else
    echo "   ✅ DPO model already cached"
fi

echo ""
echo "✅ Models ready"

# ── Update model paths in inference.py via env vars ───────────────────────────
export QLORA_MODEL_PATH="$QLORA_PATH"
export DPO_MODEL_PATH="$DPO_PATH"

# ── Launch FastAPI on port 7860 ────────────────────────────────────────────────
echo ""
echo "🚀 Starting FastAPI server..."
echo "   URL:      http://localhost:7860"
echo "   Docs:     http://localhost:7860/docs"
echo "   Frontend: http://localhost:7860/app"
echo ""

exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 1 \
    --timeout-keep-alive 30 \
    --forwarded-allow-ips='*' \
    --proxy-headers