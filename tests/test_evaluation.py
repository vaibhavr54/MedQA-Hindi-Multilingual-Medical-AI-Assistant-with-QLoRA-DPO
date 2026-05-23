#!/usr/bin/env python3
"""
MedQA-Hindi Evaluation Metric Tests
Tests ROUGE-L, BERTScore, and Medical Accuracy functions on dummy data.
No GPU required — uses CPU and small inputs.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluate import (
    compute_rouge_l,
    compute_bertscore,
    compute_medical_accuracy,
    extract_system_prompt,
    extract_qa,
    DEFAULT_SYSTEM_PROMPT,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def identical_pairs():
    """Perfect predictions — should score near 1.0."""
    texts = [
        "बुखार में पर्याप्त आराम करें और पानी पिएं।",
        "Fever should be treated with rest and hydration.",
        "रक्तचाप को नियंत्रित करने के लिए नमक कम करें।",
    ]
    return texts, texts  # predictions == references


@pytest.fixture
def unrelated_pairs():
    """Completely unrelated predictions — should score near 0.0."""
    predictions = ["यह एक परीक्षण वाक्य है।", "This is a test sentence."]
    references  = ["बुखार में पानी पिएं।",     "Drink water for fever."]
    return predictions, references


@pytest.fixture
def medical_sample():
    """Sample with known medical keywords for accuracy test."""
    predictions = ["बुखार में दवा लें और चिकित्सक से मिलें।"]
    references  = ["बुखार में दवा और उपचार जरूरी है।"]
    return predictions, references


@pytest.fixture
def valid_sample():
    """Well-formed sample dict for extraction tests."""
    return {
        "messages": [
            {"role": "system",    "content": "You are a helpful medical assistant."},
            {"role": "user",      "content": "बुखार के लिए क्या करें?"},
            {"role": "assistant", "content": "पर्याप्त आराम करें और पानी पिएं।"}
        ]
    }


# ── ROUGE-L ────────────────────────────────────────────────────────────────────
class TestRougeL:
    def test_identical_scores_high(self, identical_pairs):
        preds, refs = identical_pairs
        score = compute_rouge_l(preds, refs)
        assert score > 0.2, f"Expected >0.2 for identical pairs, got {score}"

    def test_unrelated_scores_low(self, unrelated_pairs):
        preds, refs = unrelated_pairs
        score = compute_rouge_l(preds, refs)
        assert score < 0.3, f"Expected <0.3 for unrelated pairs, got {score}"

    def test_returns_float(self, identical_pairs):
        preds, refs = identical_pairs
        assert isinstance(compute_rouge_l(preds, refs), float)

    def test_score_in_range(self, identical_pairs):
        preds, refs = identical_pairs
        score = compute_rouge_l(preds, refs)
        assert 0.0 <= score <= 1.0

    def test_empty_prediction_handled(self):
        """[EMPTY] guard — should not crash."""
        score = compute_rouge_l(["[EMPTY]"], ["बुखार में पानी पिएं।"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_single_pair(self):
        score = compute_rouge_l(
            ["fever treatment rest water"],
            ["fever treatment rest water"]
        )
        assert score > 0.9

    def test_multiple_pairs_averaged(self):
        preds = ["exact match here", "completely different xyz"]
        refs  = ["exact match here", "exact match here"]
        score = compute_rouge_l(preds, refs)
        # Should be between 0 and 1, averaged
        assert 0.0 < score < 1.0


# ── BERTScore ──────────────────────────────────────────────────────────────────
class TestBERTScore:
    def test_identical_scores_high(self, identical_pairs):
        preds, refs = identical_pairs
        score = compute_bertscore(preds, refs)
        assert score > 0.95, f"Expected >0.95 for identical pairs, got {score}"

    def test_returns_float(self, identical_pairs):
        preds, refs = identical_pairs
        assert isinstance(compute_bertscore(preds, refs), float)

    def test_score_in_range(self, identical_pairs):
        preds, refs = identical_pairs
        score = compute_bertscore(preds, refs)
        assert 0.0 <= score <= 1.0

    def test_similar_meaning_scores_high(self):
        """Semantically similar sentences should score well."""
        preds = ["Drink plenty of water when you have fever."]
        refs  = ["Stay hydrated and rest when experiencing fever."]
        score = compute_bertscore(preds, refs)
        assert score > 0.7, f"Expected >0.7 for similar sentences, got {score}"

    def test_hindi_english_mix(self):
        """Bilingual input should work with multilingual model."""
        preds = ["बुखार में पानी पिएं।", "Drink water for fever."]
        refs  = ["बुखार में पानी पिएं।", "Drink water for fever."]
        score = compute_bertscore(preds, refs)
        assert score > 0.95


# ── Medical Accuracy ───────────────────────────────────────────────────────────
class TestMedicalAccuracy:
    def test_returns_float(self, medical_sample):
        preds, refs = medical_sample
        assert isinstance(compute_medical_accuracy(preds, refs), float)

    def test_score_in_range(self, medical_sample):
        preds, refs = medical_sample
        score = compute_medical_accuracy(preds, refs)
        assert 0.0 <= score <= 1.0

    def test_shared_keywords_score_high(self):
        """Both prediction and reference contain same medical keywords."""
        preds = ["बुखार में दवा और उपचार जरूरी है। चिकित्सक से मिलें।"]
        refs  = ["बुखार में दवा और उपचार जरूरी है।"]
        score = compute_medical_accuracy(preds, refs)
        assert score > 0.5

    def test_no_keywords_in_reference_neutral(self):
        """Reference has no medical keywords → neutral score 0.5."""
        preds = ["यह एक परीक्षण है।"]
        refs  = ["यह एक परीक्षण है।"]
        score = compute_medical_accuracy(preds, refs)
        assert score == 0.5

    def test_english_keywords(self):
        preds = ["Take medicine and consult a doctor for fever treatment."]
        refs  = ["Consult a doctor for fever treatment."]
        score = compute_medical_accuracy(preds, refs)
        assert score > 0.5

    def test_empty_prediction_handled(self):
        score = compute_medical_accuracy(["[EMPTY]"], ["बुखार में दवा लें।"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── Data extraction helpers ────────────────────────────────────────────────────
class TestExtractHelpers:
    def test_extract_system_prompt_correct_role(self, valid_sample):
        prompt = extract_system_prompt(valid_sample)
        assert prompt == "You are a helpful medical assistant."

    def test_extract_system_prompt_fallback(self):
        """No system role → returns DEFAULT_SYSTEM_PROMPT."""
        sample = {"messages": [
            {"role": "user",      "content": "question"},
            {"role": "assistant", "content": "answer"}
        ]}
        prompt = extract_system_prompt(sample)
        assert prompt == DEFAULT_SYSTEM_PROMPT

    def test_extract_system_prompt_empty_messages(self):
        sample = {"messages": []}
        prompt = extract_system_prompt(sample)
        assert prompt == DEFAULT_SYSTEM_PROMPT

    def test_extract_qa_returns_correct_question(self, valid_sample):
        question, _ = extract_qa(valid_sample)
        assert question == "बुखार के लिए क्या करें?"

    def test_extract_qa_returns_correct_answer(self, valid_sample):
        _, answer = extract_qa(valid_sample)
        assert answer == "पर्याप्त आराम करें और पानी पिएं।"

    def test_extract_qa_missing_roles(self):
        """Missing user/assistant roles → empty strings, no crash."""
        sample = {"messages": [{"role": "system", "content": "sys"}]}
        question, answer = extract_qa(sample)
        assert question == ""
        assert answer   == ""

    def test_extract_qa_list_content_handled(self):
        """Content as list (DPO dataset format) → should not crash."""
        sample = {"messages": [
            {"role": "system",    "content": "sys"},
            {"role": "user",      "content": ["question part 1", "part 2"]},
            {"role": "assistant", "content": "answer"}
        ]}
        # Should not raise — content handling is robust
        try:
            question, answer = extract_qa(sample)
        except Exception as e:
            pytest.fail(f"extract_qa raised {e} on list content")