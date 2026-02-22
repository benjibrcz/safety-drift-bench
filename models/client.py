"""Model client for DriftBench.

Supports two backends:
  - "local": HuggingFace transformers (AutoTokenizer + AutoModelForCausalLM)
  - "api":   OpenAI-compatible API (e.g. OpenRouter, vLLM, Together)

Includes optional SQLite disk-cache keyed on (model, params, full prompt).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set Python / NumPy / Torch RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(model_name: str, params: dict, messages: list[dict]) -> str:
    payload = json.dumps(
        {"model": model_name, "params": params, "messages": messages},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class DiskCache:
    """Thin SQLite cache for model responses."""

    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._db.commit()

    def get(self, key: str) -> Optional[str]:
        row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------

class ModelClient(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


# ---------------------------------------------------------------------------
# Local HuggingFace client
# ---------------------------------------------------------------------------

class LocalHFClient:
    """Run inference locally via HuggingFace transformers."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 200,
        cache_path: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        torch_dtype = torch.float16 if dtype == "auto" else getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        )
        self.cache = DiskCache(cache_path) if cache_path else None

    def generate(self, messages: list[dict]) -> str:
        import torch

        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }

        if self.cache:
            key = _cache_key(self.model_name, params, messages)
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        # Apply chat template (falls back to simple concat if unavailable)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant: "

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        if self.cache:
            self.cache.put(key, response)

        return response


# ---------------------------------------------------------------------------
# API client (OpenAI-compatible — works with OpenRouter, vLLM, Together, etc.)
# ---------------------------------------------------------------------------

class APIClient:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 200,
        cache_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)
        self.cache = DiskCache(cache_path) if cache_path else None

    def generate(self, messages: list[dict]) -> str:
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }

        if self.cache:
            key = _cache_key(self.model_name, params, messages)
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        result = resp.choices[0].message.content.strip()

        if self.cache:
            self.cache.put(key, result)

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(config: dict) -> ModelClient:
    """Create the appropriate client from a config dict (model section)."""
    model_cfg = config.get("model", config)
    sampling = config.get("sampling", {})
    cache_path = config.get("cache_path")
    backend = model_cfg.get("backend", "local")

    if backend == "api":
        return APIClient(
            model_name=model_cfg["model_name"],
            temperature=sampling.get("temperature", 0.7),
            top_p=sampling.get("top_p", 0.9),
            max_new_tokens=sampling.get("max_new_tokens", 200),
            cache_path=cache_path,
            api_key=model_cfg.get("api_key"),
            base_url=model_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        )
    else:
        return LocalHFClient(
            model_name=model_cfg["model_name"],
            device=model_cfg.get("device", "auto"),
            dtype=model_cfg.get("dtype", "auto"),
            temperature=sampling.get("temperature", 0.7),
            top_p=sampling.get("top_p", 0.9),
            max_new_tokens=sampling.get("max_new_tokens", 200),
            cache_path=cache_path,
        )
