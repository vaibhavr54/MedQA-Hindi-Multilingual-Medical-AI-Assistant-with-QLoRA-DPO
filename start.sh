#!/bin/bash
# MedQA-Hindi Quick Start Script

echo "========================================"
echo "🏥 MedQA-Hindi Setup & Launch"
echo "========================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📌 Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Check if models are trained
if [ -d "outputs/dpo/final_merged" ]; then
    echo ""
    echo "✅ DPO model found"
    MODEL_STATUS="trained"
elif [ -d "outputs/qlora/final_merged" ]; then
    echo ""
    echo "⚠️  Only QLoRA model found (DPO not trained)"
    MODEL_STATUS="partial"
else
    echo ""
    echo "⚠️  No trained models found. Using base model only."
    echo "   Run training pipeline first:"
    echo "   1. python data/prepare_dataset.py"
    echo "   2. python training/qlora_finetune.py"
    echo "   3. python training/dpo_align.py"
    MODEL_STATUS="base_only"
fi

# Start API
echo ""
echo "🚀 Starting FastAPI server..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Frontend: http://localhost:8000/app"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
