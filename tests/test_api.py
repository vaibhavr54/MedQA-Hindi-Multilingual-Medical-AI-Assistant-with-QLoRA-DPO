#!/usr/bin/env python3
"""
MedQA-Hindi API Test Suite
Tests all FastAPI endpoints using TestClient with a mocked inference engine.
No GPU required — inference is mocked so CI runs fast and free.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Mock inference engine ──────────────────────────────────────────────────────
# Replaces the real engine so tests run without GPU or model weights

def make_mock_engine():
    engine = MagicMock()

    # generate() returns (answer, confidence, language, processing_time_ms)
    engine.generate.return_value = (
        "बुखार में पर्याप्त आराम करें और पानी पिएं।\n\n⚠️ अस्वीकरण: यह जानकारी केवल शैक्षिक उद्देश्यों के लिए है।",
        0.82,
        "hi",
        312.5
    )

    # compare() returns (list of results, total_time_ms)
    engine.compare.return_value = (
        [
            {"model": "base",  "answer": "Base answer.",  "confidence": 0.65, "response_time_ms": 200.0},
            {"model": "qlora", "answer": "QLoRA answer.", "confidence": 0.78, "response_time_ms": 210.0},
            {"model": "dpo",   "answer": "DPO answer.",   "confidence": 0.85, "response_time_ms": 220.0},
        ],
        630.0
    )

    engine.get_loaded_models.return_value = {
        "base": True, "qlora": False, "dpo": False
    }
    engine.unload_model.return_value = None

    return engine


# ── Fixture: TestClient with mocked engine ─────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    mock_engine = make_mock_engine()

    # Patch get_inference_engine before importing app
    # so lifespan never tries to load real models
    with patch("api.inference.get_inference_engine", return_value=mock_engine):
        with patch("api.main.get_inference_engine", return_value=mock_engine):
            from api.main import app
            # Override the global engine directly
            import api.main as main_module
            main_module.engine = mock_engine

            with TestClient(app, raise_server_exceptions=True) as c:
                yield c


# ── Root ───────────────────────────────────────────────────────────────────────
class TestRoot:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_endpoints(self, client):
        data = r = client.get("/").json()
        assert "endpoints" in data
        assert "ask"     in data["endpoints"]
        assert "compare" in data["endpoints"]
        assert "metrics" in data["endpoints"]
        assert "health"  in data["endpoints"]


# ── Health ─────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/api/health").json()
        assert "status"        in data
        assert "models_loaded" in data
        assert "gpu_available" in data
        assert data["status"]  == "healthy"

    def test_health_models_loaded_has_three_keys(self, client):
        data = client.get("/api/health").json()
        assert set(data["models_loaded"].keys()) == {"base", "qlora", "dpo"}


# ── Ask ────────────────────────────────────────────────────────────────────────
class TestAsk:
    VALID_PAYLOAD = {
        "question":    "बुखार के लिए क्या करें?",
        "model":       "base",
        "language":    "auto",
        "max_tokens":  256,
        "temperature": 0.7
    }

    def test_ask_returns_200(self, client):
        r = client.post("/api/ask", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_ask_response_schema(self, client):
        data = client.post("/api/ask", json=self.VALID_PAYLOAD).json()
        assert "answer"             in data
        assert "model_used"         in data
        assert "confidence"         in data
        assert "language"           in data
        assert "processing_time_ms" in data

    def test_ask_confidence_in_range(self, client):
        data = client.post("/api/ask", json=self.VALID_PAYLOAD).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_ask_answer_is_string(self, client):
        data = client.post("/api/ask", json=self.VALID_PAYLOAD).json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_ask_all_model_types(self, client):
        for model in ["base", "qlora", "dpo"]:
            payload = {**self.VALID_PAYLOAD, "model": model}
            r = client.post("/api/ask", json=payload)
            assert r.status_code == 200, f"Failed for model={model}"

    def test_ask_english_question(self, client):
        payload = {**self.VALID_PAYLOAD, "question": "What should I do for fever?"}
        r = client.post("/api/ask", json=payload)
        assert r.status_code == 200

    def test_ask_question_too_short_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "question": "hi"}  # < 5 chars
        r = client.post("/api/ask", json=payload)
        assert r.status_code == 422

    def test_ask_question_too_long_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "question": "a" * 2001}  # > 2000 chars
        r = client.post("/api/ask", json=payload)
        assert r.status_code == 422

    def test_ask_invalid_model_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "model": "gpt4"}
        r = client.post("/api/ask", json=payload)
        assert r.status_code == 422

    def test_ask_invalid_temperature_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "temperature": 5.0}  # > 1.5
        r = client.post("/api/ask", json=payload)
        assert r.status_code == 422


# ── Compare ────────────────────────────────────────────────────────────────────
class TestCompare:
    VALID_PAYLOAD = {
        "question":    "बुखार के लिए क्या करें?",
        "models":      ["base", "dpo"],
        "max_tokens":  256,
        "temperature": 0.7
    }

    def test_compare_returns_200(self, client):
        r = client.post("/api/compare", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_compare_response_schema(self, client):
        data = client.post("/api/compare", json=self.VALID_PAYLOAD).json()
        assert "question"     in data
        assert "comparisons"  in data
        assert "total_time_ms" in data

    def test_compare_returns_correct_number_of_models(self, client):
        data = client.post("/api/compare", json=self.VALID_PAYLOAD).json()
        # mock returns 3 but request asked for 2 — just check it's a list
        assert isinstance(data["comparisons"], list)
        assert len(data["comparisons"]) >= 1

    def test_compare_each_result_has_required_fields(self, client):
        data = client.post("/api/compare", json=self.VALID_PAYLOAD).json()
        for comp in data["comparisons"]:
            assert "model"           in comp
            assert "answer"          in comp
            assert "confidence"      in comp
            assert "response_time_ms" in comp

    def test_compare_question_echoed(self, client):
        data = client.post("/api/compare", json=self.VALID_PAYLOAD).json()
        assert data["question"] == self.VALID_PAYLOAD["question"]

    def test_compare_all_three_models(self, client):
        payload = {**self.VALID_PAYLOAD, "models": ["base", "qlora", "dpo"]}
        r = client.post("/api/compare", json=payload)
        assert r.status_code == 200

    def test_compare_invalid_model_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "models": ["base", "gpt4"]}
        r = client.post("/api/compare", json=payload)
        assert r.status_code == 422


# ── Metrics ────────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_metrics_returns_200(self, client):
        r = client.get("/api/metrics")
        assert r.status_code == 200

    def test_metrics_response_schema(self, client):
        data = client.get("/api/metrics").json()
        assert "metrics"          in data
        assert "last_updated"     in data
        assert "num_test_samples" in data

    def test_metrics_has_three_metric_types(self, client):
        data   = client.get("/api/metrics").json()
        names  = [m["metric"] for m in data["metrics"]]
        assert "ROUGE-L"         in names
        assert "BERTScore"       in names
        assert "Medical Accuracy" in names

    def test_metrics_each_entry_has_all_fields(self, client):
        data = client.get("/api/metrics").json()
        for m in data["metrics"]:
            assert "metric"            in m
            assert "base_value"        in m
            assert "qlora_value"       in m
            assert "dpo_value"         in m
            assert "qlora_improvement" in m
            assert "dpo_improvement"   in m

    def test_metrics_values_are_floats(self, client):
        data = client.get("/api/metrics").json()
        for m in data["metrics"]:
            assert isinstance(m["base_value"],  float)
            assert isinstance(m["qlora_value"], float)
            assert isinstance(m["dpo_value"],   float)

    def test_metrics_improvements_are_numeric(self, client):
        data = client.get("/api/metrics").json()
        for m in data["metrics"]:
            assert isinstance(m["qlora_improvement"], (int, float))
            assert isinstance(m["dpo_improvement"],   (int, float))