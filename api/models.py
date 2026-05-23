#!/usr/bin/env python3
"""
MedQA-Hindi API Pydantic Models
Request/response schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Available model variants."""
    BASE = "base"
    QLORA = "qlora"
    DPO = "dpo"


class Language(str, Enum):
    """Supported languages."""
    AUTO = "auto"
    HINDI = "hi"
    ENGLISH = "en"


class AskRequest(BaseModel):
    """Request schema for /api/ask endpoint."""
    question: str = Field(
        ..., 
        min_length=5, 
        max_length=2000,
        description="Medical question to ask the model"
    )
    model: ModelType = Field(
        default=ModelType.DPO,
        description="Which model variant to use"
    )
    language: Language = Field(
        default=Language.AUTO,
        description="Response language (auto-detects if not specified)"
    )
    max_tokens: int = Field(
        default=512,
        ge=50,
        le=2048,
        description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.5,
        description="Sampling temperature"
    )


class AskResponse(BaseModel):
    """Response schema for /api/ask endpoint."""
    answer: str = Field(..., description="Generated medical answer")
    model_used: str = Field(..., description="Model variant used")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score (based on generation probability)"
    )
    language: str = Field(..., description="Detected response language")
    disclaimer: str = Field(
        default="यह जानकारी केवल शैक्षिक उद्देश्यों के लिए है। यह पेशेवर चिकित्सा सलाह का विकल्प नहीं है।",
        description="Medical disclaimer"
    )
    processing_time_ms: float = Field(..., description="Request processing time in milliseconds")


class CompareRequest(BaseModel):
    """Request schema for /api/compare endpoint."""
    question: str = Field(
        ..., 
        min_length=5, 
        max_length=2000,
        description="Medical question to compare across models"
    )
    models: List[ModelType] = Field(
        default=[ModelType.BASE, ModelType.DPO],
        description="List of models to compare"
    )
    max_tokens: int = Field(default=512, ge=50, le=2048)
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)


class ModelComparison(BaseModel):
    """Single model comparison result."""
    model: str = Field(..., description="Model name")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0)
    response_time_ms: float = Field(..., description="Individual model response time")


class CompareResponse(BaseModel):
    """Response schema for /api/compare endpoint."""
    question: str = Field(..., description="Original question")
    comparisons: List[ModelComparison] = Field(..., description="Results from each model")
    total_time_ms: float = Field(..., description="Total processing time")


class MetricsData(BaseModel):
    """Single metric data point."""
    metric: str = Field(..., description="Metric name")
    base_value: float = Field(..., description="Base model score")
    qlora_value: float = Field(..., description="QLoRA model score")
    dpo_value: float = Field(..., description="DPO model score")
    qlora_improvement: float = Field(..., description="QLoRA improvement %")
    dpo_improvement: float = Field(..., description="DPO improvement %")


class MetricsResponse(BaseModel):
    """Response schema for /api/metrics endpoint."""
    metrics: List[MetricsData] = Field(..., description="All evaluation metrics")
    last_updated: str = Field(..., description="Last evaluation timestamp")
    num_test_samples: int = Field(..., description="Number of test samples used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    models_loaded: Dict[str, bool] = Field(..., description="Which models are loaded")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_memory_gb: Optional[float] = Field(None, description="Available GPU memory")
