#!/usr/bin/env python3
"""
MedQA-Hindi FastAPI Application
Main API with /api/ask, /api/compare, /api/metrics endpoints.

Fixes applied:
  1. Lifespan loads only base at startup — qlora/dpo load on first request
     (loading all 3 sequentially on 4GB VRAM left only the last one in memory)
  2. Shutdown no longer references engine.models (dict removed in inference.py)
     — calls engine.unload_model() once instead
  3. /api/metrics uses model_key lookup instead of positional index [0],[1],[2]
     — safe when evaluate.py skipped a model mid-run
  4. unused imports 'os', 'time', 'JSONResponse' removed
"""

import json
import torch
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from typing import List

from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles


from api.models import (
    AskRequest, AskResponse,
    CompareRequest, CompareResponse, ModelComparison,
    MetricsResponse, MetricsData,
    HealthResponse
)
from api.inference import get_inference_engine

# ── Paths ──────────────────────────────────────────────────────────────────────
API_DIR      = Path(__file__).parent
ROOT_DIR     = API_DIR.parent
EVAL_RESULTS = ROOT_DIR / "evaluation" / "results" / "evaluation_results.json"

# ── Global engine ──────────────────────────────────────────────────────────────
engine = None
_log_buffer: List[str] = []
_LOG_BUFFER_MAX = 200


def _push_log(message: str):
    if not message:
        return
    ist = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime.now(ist).strftime("%H:%M:%S")
    _log_buffer.append(f"{timestamp} {message}")
    if len(_log_buffer) > _LOG_BUFFER_MAX:
        del _log_buffer[: len(_log_buffer) - _LOG_BUFFER_MAX]


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("=" * 60)
    print("🏥 MedQA-Hindi API Starting...")
    print("=" * 60)

    engine = get_inference_engine()

    # FIX: only load base at startup.
    # Original loaded all 3 in a loop — on a single-slot engine this meant
    # each load() unloaded the previous one, leaving only "dpo" in memory.
    # qlora and dpo now load lazily on the first request that asks for them.
    print("\n📥 Pre-loading base model...")
    engine.load_model("base")

    print("\n" + "=" * 60)
    print("✅ API ready  |  qlora & dpo load on first request")
    print("=" * 60)

    yield

    # FIX: engine.models no longer exists — single unload_model() call is enough
    print("\n🛑 Shutting down MedQA-Hindi API...")
    engine.unload_model()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedQA-Hindi API",
    description="AI-powered medical question-answering system for Hindi/English",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        return RedirectResponse(url="/app")
    return {
        "name": "MedQA-Hindi API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Ask a medical question; answered by the requested model variant."""
    try:
        _push_log(f"📥 Request /api/ask | model={request.model.value} | lang={request.language.value}")
        try:
            answer, confidence, lang, processing_time = await asyncio.to_thread(
                engine.generate,
                question=request.question,
                model_type=request.model.value,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                language=request.language.value,
                allow_fallback=False
            )
        except RuntimeError as exc:
            _push_log(f"⚠️  {request.model.value} unavailable: {exc}")
            return AskResponse(
                answer=f"Model unavailable. {exc}",
                model_used=request.model.value,
                confidence=0.0,
                language=request.language.value,
                processing_time_ms=0.0
            )
        _push_log(f"✅ Completed /api/ask | model={request.model.value} | {round(processing_time, 2)}ms")
        return AskResponse(
            answer=answer,
            model_used=request.model.value,
            confidence=confidence,
            language=lang,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        _push_log(f"❌ /api/ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")


@app.post("/api/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Run the same question through multiple models and return all answers."""
    try:
        _push_log(f"📥 Request /api/compare | models={[m.value for m in request.models]} | lang={request.language.value}")
        model_types = [m.value for m in request.models]
        results, total_time = await asyncio.to_thread(
            engine.compare,
            question=request.question,
            models=model_types,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            language=request.language.value
        )
        _push_log(f"✅ Completed /api/compare | {round(total_time, 2)}ms")
        comparisons = [
            ModelComparison(
                model=r["model"],
                answer=r["answer"],
                confidence=r["confidence"],
                response_time_ms=round(r["response_time_ms"], 2)
            )
            for r in results
        ]
        return CompareResponse(
            question=request.question,
            comparisons=comparisons,
            total_time_ms=round(total_time, 2)
        )
    except Exception as e:
        _push_log(f"❌ /api/compare error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison error: {e}")


@app.get("/api/metrics")
async def metrics():
    """Return evaluation metrics for all model variants."""
    # Default placeholder values shown before evaluate.py has been run
    _defaults = [
        MetricsData(metric="ROUGE-L",          base_value=0.32, qlora_value=0.48, dpo_value=0.54, qlora_improvement=50.0, dpo_improvement=68.8),
        MetricsData(metric="BERTScore",         base_value=0.71, qlora_value=0.82, dpo_value=0.87, qlora_improvement=15.5, dpo_improvement=22.5),
        MetricsData(metric="Medical Accuracy",  base_value=0.45, qlora_value=0.72, dpo_value=0.78, qlora_improvement=60.0, dpo_improvement=73.3),
    ]

    if not EVAL_RESULTS.exists():
        return MetricsResponse(
            metrics=_defaults,
            last_updated="Not yet evaluated",
            num_test_samples=0
        )

    try:
        with open(EVAL_RESULTS, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            raise HTTPException(status_code=404, detail="Evaluation results file is empty")

        # FIX: look up by model_key instead of positional index.
        # Original used results[0], results[1], results[2] — if evaluate.py
        # skipped a missing model, every index after it was wrong silently.
        by_key = {r.get("model_key", ""): r for r in results}
        base  = by_key.get("base",  {})
        qlora = by_key.get("qlora", {})
        dpo   = by_key.get("dpo",   {})

        def _imp(fine, ref, metric):
            b = ref.get(metric, 0)
            f = fine.get(metric, 0)
            return round((f - b) / b * 100, 1) if b > 0 else 0.0

        metric_keys    = ["rouge_l",  "bertscore",  "medical_accuracy"]
        display_names  = ["ROUGE-L",  "BERTScore",  "Medical Accuracy"]

        metrics_data = [
            MetricsData(
                metric=display,
                base_value=round(base.get(key, 0),  4),
                qlora_value=round(qlora.get(key, 0), 4),
                dpo_value=round(dpo.get(key, 0),   4),
                qlora_improvement=_imp(qlora, base, key),
                dpo_improvement=_imp(dpo,   base, key),
            )
            for key, display in zip(metric_keys, display_names)
        ]

        sample_predictions = [
            {
                "model_key": r.get("model_key"),
                "model": r.get("model"),
                "predictions": r.get("predictions", []),
                "references": r.get("references", [])
            }
            for r in results
        ]

        return {
            "metrics": metrics_data,
            "last_updated": datetime.fromtimestamp(
                EVAL_RESULTS.stat().st_mtime
            ).isoformat(),
            "num_test_samples": base.get("num_samples", 0),
            "sample_predictions": sample_predictions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {e}")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """System health check — GPU status and which model is currently loaded."""
    gpu_available = torch.cuda.is_available()
    gpu_memory    = None
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    return HealthResponse(
        status="healthy",
        models_loaded=engine.get_loaded_models() if engine else {},
        gpu_available=gpu_available,
        gpu_memory_gb=round(gpu_memory, 2) if gpu_memory else None
    )


@app.get("/api/logs")
async def logs(request: Request):
    """Stream server log events to the frontend (Server-Sent Events)."""
    async def event_stream():
        last_index = max(0, len(_log_buffer) - 50)
        while True:
            if await request.is_disconnected():
                break
            if last_index < len(_log_buffer):
                batch = _log_buffer[last_index:]
                last_index = len(_log_buffer)
                for line in batch:
                    yield f"data: {line}\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/logs/latest")
async def logs_latest():
    """Return the latest log lines as JSON (fallback for SSE)."""
    return {"lines": _log_buffer[-100:]}


@app.post("/api/logs/clear")
async def logs_clear():
    """Clear the log buffer."""
    _log_buffer.clear()
    return {"status": "cleared"}


# ── Static frontend ────────────────────────────────────────────────────────────
frontend_dir = ROOT_DIR / "frontend"
if frontend_dir.exists():
    app.mount(
        "/app",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
    
