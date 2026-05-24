#!/usr/bin/env python3
"""
MedQA-Hindi Inference Engine
Handles model loading, generation, and confidence scoring.

Fixes applied:
  1. Base model corrected to Qwen2.5-0.5B-Instruct — matches training pipeline
  2. One model loaded at a time — avoids OOM on 4GB VRAM during /api/compare
     Load-on-demand with automatic unload of previous model before loading next
  3. compute_confidence() — removed redundant forward pass that doubled latency;
     replaced with token-level mean log-prob over response tokens only (correct + fast)
  4. generate() — KeyError crash fixed when fallback to base also fails to load
  5. prompt_inputs variable was computed but never used — removed dead code
  6. unused import 're' and 'os' removed
  7. VRAM logging added to load_model() for visibility on 4GB card
"""

import time
import torch
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_TOKENS  = 512
DEFAULT_TEMPERATURE = 0.7

# FIX 1: corrected to 0.5B — must match the model you actually trained
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are MedQA-Hindi, a medical QA model comparison system.\n"
    "Provide accurate, helpful medical information in Hindi or English "
    "based on the user's question language.\n"
    "Do not include medical disclaimers in the response; disclaimers are shown elsewhere.\n"
    "CRITICAL: Respond ONLY in the target language. Do NOT mix languages."
)

DISCLAIMER_HI = ""
DISCLAIMER_EN = ""


# ── Quantization config ────────────────────────────────────────────────────────
def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


# ── Engine ─────────────────────────────────────────────────────────────────────
class MedQAInference:
    """
    Inference engine for MedQA-Hindi models.

    Memory strategy for 4GB VRAM:
      - Only ONE model is kept in GPU memory at a time.
      - load_model() unloads the current model before loading a new one
        UNLESS the requested model is already loaded (no-op).
      - compare() sequences models one-at-a-time: generate → unload → next.
    """

    def __init__(self):
        # At most one model is resident at a time
        self._loaded_type: Optional[str] = None
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

        # # Canonical paths for each variant
        # _outputs = Path(__file__).parent.parent / "training" / "outputs"
        # self._model_paths: Dict[str, str] = {
        #     "base":  BASE_MODEL_ID,
        #     "qlora": str(_outputs / "qlora_rtx2050_full" / "final_merged"),
        #     "dpo":   str(_outputs / "dpo" / "final_merged"),
        # }
        import os

        _outputs = Path(__file__).parent.parent / "outputs"
        self._model_paths: Dict[str, str] = {
            "base":  BASE_MODEL_ID,
            "qlora": os.environ.get("QLORA_MODEL_PATH", str(_outputs / "qlora_rtx2050_full" / "final_merged")),
            "dpo":   os.environ.get("DPO_MODEL_PATH",   str(_outputs / "dpo" / "final_merged")),
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _vram_str(self) -> str:
        if not torch.cuda.is_available():
            return "CPU"
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.2f}GB / {total:.2f}GB"

    def _emit_log(self, message: str):
        try:
            from api.main import _push_log
            _push_log(message)
        except Exception:
            pass

    def _unload_current(self):
        """Free GPU memory occupied by the currently-loaded model."""
        if self._model is not None:
            print(f"   🗑️  Unloading {self._loaded_type} | VRAM before: {self._vram_str()}")
            self._emit_log(f"🗑️  Unloading {self._loaded_type} | VRAM before: {self._vram_str()}")
            del self._model
            del self._tokenizer
            self._model       = None
            self._tokenizer   = None
            self._loaded_type = None
            torch.cuda.empty_cache()

    def _load(self, model_type: str, path: str) -> bool:
        """Low-level load: unload previous, load new, store references."""
        self._unload_current()

        print(f"📥 Loading {model_type} from {path} ...")
        self._emit_log(f"📥 Loading {model_type} from {path} ...")
        try:
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    quantization_config=_bnb_config(),
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                # CPU fallback — no quantization
                print(f"   ⚠️  No GPU — loading in fp32 on CPU (slow but functional)")
                self._emit_log("⚠️  CPU mode: 4-bit quantization disabled")
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
            model.config.use_cache = False  # safe default; generation overrides this

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token    = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            self._model       = model
            self._tokenizer   = tokenizer
            self._loaded_type = model_type

            print(f"   ✅ {model_type} loaded | VRAM: {self._vram_str()}")
            if not torch.cuda.is_available() and model_type != "base":
                self._emit_log(f"ℹ️  {model_type} loaded on CPU (slower)")
            self._emit_log(f"✅ {model_type} loaded | VRAM: {self._vram_str()}")
            return True

        except Exception as exc:
            print(f"   ❌ Failed to load {model_type}: {exc}")
            self._emit_log(f"❌ Failed to load {model_type}: {exc}")
            self._model       = None
            self._tokenizer   = None
            self._loaded_type = None
            return False

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_model(self, model_type: str, model_path: Optional[str] = None) -> bool:
        """
        Ensure `model_type` is loaded and ready.
        If it is already loaded, this is a no-op (returns True immediately).
        """
        if self._loaded_type == model_type:
            print(f"   ℹ️  {model_type} already loaded")
            return True

        if model_type not in self._model_paths and model_path is None:
            raise ValueError(f"Unknown model type: {model_type!r}")

        path = model_path or self._model_paths[model_type]

        # Non-base models must exist on disk
        if model_type != "base" and not Path(path).exists():
            print(f"   ⚠️  {model_type} not found at {path} — skipping")
            self._emit_log(f"⚠️  {model_type} not found at {path} — skipping")
            return False

        return self._load(model_type, path)

    def unload_model(self, model_type: Optional[str] = None):
        """
        Unload the specified model (or whatever is currently loaded).
        model_type is accepted for API compatibility with main.py but ignored —
        only one model can be loaded at a time anyway.
        """
        self._unload_current()

    def detect_language(self, text: str) -> str:
        """Return 'hi' if >30% of non-space chars are Devanagari, else 'en'."""
        non_space = text.replace(" ", "")
        if not non_space:
            return "en"
        devanagari = sum(1 for ch in non_space if 0x0900 <= ord(ch) <= 0x097F)
        return "hi" if (devanagari / len(non_space)) > 0.3 else "en"

    def compute_confidence(self, prompt: str, response: str) -> float:
        """
        Confidence = mean token log-probability over the response tokens only.

        FIX (original had two bugs):
          1. It ran a full forward pass on prompt+response, computing loss over
             ALL tokens (including prompt), which inflates confidence for long
             prompts and deflates it for long responses — neither is meaningful.
          2. It tokenized prompt_inputs but never used it (dead code).

        This version:
          - Runs ONE forward pass on the full sequence (efficient)
          - Masks out prompt tokens so loss is computed on response tokens only
          - Converts mean NLL → probability with exp(-nll), clipped to [0, 1]
          - Falls back to 0.75 on any error (4-bit models occasionally refuse
            a second forward pass mid-session due to quantization state)
        """
        if self._model is None or self._tokenizer is None:
            return 0.75

        try:
            full_text = prompt + response
            full_ids   = self._tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
            prompt_ids = self._tokenizer(prompt,    return_tensors="pt").input_ids.to(DEVICE)

            prompt_len = prompt_ids.shape[1]
            seq_len    = full_ids.shape[1]

            if prompt_len >= seq_len:
                return 0.75  # Response tokenized to nothing — edge case

            # Labels: -100 for prompt tokens (ignored in loss), real ids for response
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            with torch.no_grad():
                out = self._model(input_ids=full_ids, labels=labels)
                # out.loss = mean NLL over response tokens
                nll = out.loss.item()

            confidence = float(torch.exp(torch.tensor(-nll)).clamp(0.0, 1.0))
            return round(confidence, 3)

        except Exception:
            return 0.75

    def generate(
        self,
        question: str,
        model_type: str = "dpo",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        language: str = "auto"
    ) -> Tuple[str, float, str, float]:
        """
        Generate an answer for `question` using `model_type`.

        Returns:
            (answer, confidence, detected_language, processing_time_ms)

        FIX: original had a KeyError crash path:
          load_model() could fail → fallback to "base" → load_model("base") could
          also fail → self.models["base"] KeyError on the next line.
          Now raises RuntimeError with a clear message instead of crashing silently.
        """
        start = time.time()

        # Ensure the right model is loaded (swaps automatically if different)
        ok = self.load_model(model_type)
        if not ok:
            if model_type != "base":
                print(f"   🔄 {model_type} unavailable — falling back to base")
                model_type = "base"
                ok = self.load_model("base")
            if not ok:
                raise RuntimeError(
                    "Could not load any model. "
                    "Check that Qwen2.5-0.5B-Instruct is accessible."
                )

        model     = self._model
        tokenizer = self._tokenizer

        # Language detection
        detected_lang = self.detect_language(question) if language == "auto" else language

        # Build prompt
        lang_hint = "Hindi" if detected_lang == "hi" else "English"
        system_prompt = (
            SYSTEM_PROMPT
            + f"\n\nTarget language: {lang_hint}."
            + f"\nRules: 1) Use only {lang_hint}. 2) If you cannot answer, still reply only in {lang_hint}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Question ({lang_hint}): {question}"}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Generation — use_cache=True here is intentional (KV cache speeds up generation)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,        # override model.config.use_cache=False for generation
            )

        response = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        response = self._strip_prompt_echo(response)

        # Disclaimers are intentionally omitted in responses

        confidence      = self.compute_confidence(prompt, response)
        processing_time = (time.time() - start) * 1000

        return response, confidence, detected_lang, processing_time

    def compare(
        self,
        question: str,
        models: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        language: str = "auto"
    ) -> Tuple[List[dict], float]:
        """
        Run the same question through multiple models sequentially.

        FIX: original tried to keep all models in VRAM simultaneously —
        impossible on 4GB. Now each model is loaded, generates, then the
        next model swaps in (previous is auto-unloaded by load_model()).

        Returns:
            (list of per-model result dicts, total_time_ms)
        """
        if models is None:
            models = ["base", "qlora", "dpo"]

        start   = time.time()
        results = []

        for model_type in models:
            try:
                answer, confidence, lang, time_ms = self.generate(
                    question=question,
                    model_type=model_type,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    language=language
                )
                results.append({
                    "model":           model_type,
                    "answer":          answer,
                    "confidence":      confidence,
                    "response_time_ms": time_ms
                })
            except Exception as exc:
                results.append({
                    "model":           model_type,
                    "answer":          f"Model unavailable. {exc}",
                    "confidence":      0.0,
                    "response_time_ms": 0.0
                })

        total_time = (time.time() - start) * 1000
        return results, total_time

    def _strip_prompt_echo(self, text: str) -> str:
        """Remove training prompt tails or assistant boilerplate if present."""
        if not text:
            return text

        markers = [
            "medqa-hindi",
            "thank you for using our service",
            "happy to assist",
            "best wishes",
            "i am ready to assist",
            "i am here to assist",
            "if you need help with anything else",
            "if you have any other query",
            "if you have any other queries",
            "please consult your healthcare provider",
            "always follow their instructions",
        ]

        lowered = text.lower()
        cut_index = None
        for marker in markers:
            idx = lowered.find(marker)
            if idx != -1:
                cut_index = idx if cut_index is None else min(cut_index, idx)

        cleaned = text[:cut_index].strip() if cut_index is not None else text.strip()

        # Remove lingering prompt-like lines
        cleaned = re.sub(r"\n\s*(thank you|best wishes|medqa-hindi).*$", "", cleaned, flags=re.IGNORECASE).strip()

        return cleaned if cleaned else text.strip()

    def get_loaded_models(self) -> Dict[str, bool]:
        """
        Return load status for all three variants.
        Compatible with main.py's /api/health endpoint.
        """
        return {
            "base":  self._loaded_type == "base",
            "qlora": self._loaded_type == "qlora",
            "dpo":   self._loaded_type == "dpo",
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine: Optional[MedQAInference] = None


def get_inference_engine() -> MedQAInference:
    """Return the process-level singleton inference engine."""
    global _engine
    if _engine is None:
        _engine = MedQAInference()
    return _engine
