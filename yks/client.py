"""Unified async model client.

Backends (set per provider in configs/systems.yaml with `type`):
  openai_compat     Fireworks, OpenAI, Gemini, ... via the OpenAI SDK (Chat Completions)
  bedrock_converse  Amazon Bedrock Converse API via boto3 (Claude, OpenAI GPT on Bedrock, ...).
                    Uses the standard AWS credential chain (aws login / AWS_PROFILE); no keys in .env.

Billing (set per model):
  token     (default) cost = in_tokens * price_in + out_tokens * price_out   (serverless APIs)
  gpu_hour  dedicated deployment (e.g. a Fireworks fine-tuned model). Per-call cost is 0 here;
            report.py charges GPU-hours over the measured run wall time instead.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Params the OpenAI SDK accepts directly; everything else goes through extra_body
# (e.g. Fireworks' reasoning_effort, top_k) so provider-specific knobs still work.
_NATIVE = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop"}


def load_config(path: str | Path = ROOT / "configs" / "systems.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class CallResult:
    text: str | None
    latency_s: float
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class Model:
    name: str
    provider: str
    backend: str
    model: str
    price_in: float = 0.0
    price_out: float = 0.0
    price_cache_read: float | None = None   # defaults to price_in (conservative)
    price_cache_write: float | None = None  # defaults to price_in
    billing: str = "token"
    params: dict = field(default_factory=dict)
    _client: object | None = None

    @classmethod
    def from_config(cls, cfg: dict, name: str) -> "Model":
        m = cfg["models"][name]
        prov = cfg["providers"][m["provider"]]
        backend = prov.get("type", "openai_compat")
        obj = cls(name=name, provider=m["provider"], backend=backend, model=m["model"],
                  price_in=float(m.get("price_in", 0)), price_out=float(m.get("price_out", 0)),
                  billing=m.get("billing", "token"), params=dict(m.get("params") or {}),
                  price_cache_read=m.get("price_cache_read"), price_cache_write=m.get("price_cache_write"))
        if backend == "openai_compat":
            from openai import AsyncOpenAI
            key = os.environ.get(prov["api_key_env"])
            if not key:
                raise RuntimeError(f"Missing {prov['api_key_env']} for model '{name}' (set it in .env)")
            obj._client = AsyncOpenAI(base_url=prov["base_url"], api_key=key, max_retries=6, timeout=600)
        elif backend == "bedrock_converse":
            import boto3
            from botocore.config import Config
            session = boto3.Session(profile_name=prov.get("profile") or os.environ.get("AWS_PROFILE"))
            obj._client = session.client(
                "bedrock-runtime", region_name=prov.get("region", "us-east-1"),
                config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                              read_timeout=900, connect_timeout=30, max_pool_connections=64))
        else:
            raise ValueError(f"unknown provider type '{backend}'")
        return obj

    def _cost(self, fresh_in: int, tout: int, cache_read: int = 0, cache_write: int = 0) -> float:
        if self.billing != "token":
            return 0.0
        cr = self.price_in if self.price_cache_read is None else float(self.price_cache_read)
        cw = self.price_in if self.price_cache_write is None else float(self.price_cache_write)
        return (fresh_in * self.price_in + cache_read * cr + cache_write * cw + tout * self.price_out) / 1e6

    async def chat(self, messages: list[dict], **overrides) -> CallResult:
        params = {**self.params, **overrides}
        t0 = time.perf_counter()
        try:
            if self.backend == "bedrock_converse":
                text, fresh, tout, cr, cw = await self._converse(messages, params)
            else:
                text, fresh, tout, cr, cw = await self._openai(messages, params)
        except Exception as e:  # noqa: BLE001 - record and keep going; failures are reported, not hidden
            return CallResult(text=None, latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}"[:500])
        # in_tokens = ALL input tokens (fresh + cached); cost prices cached ones at the cache rate.
        return CallResult(text=text, latency_s=time.perf_counter() - t0, in_tokens=fresh + cr + cw,
                          out_tokens=tout, cost_usd=self._cost(fresh, tout, cr, cw))

    async def _openai(self, messages: list[dict], params: dict) -> tuple[str | None, int, int, int, int]:
        native = {k: v for k, v in params.items() if k in _NATIVE}
        extra = {k: v for k, v in params.items() if k not in _NATIVE}
        resp = await self._client.chat.completions.create(
            model=self.model, messages=messages, extra_body=extra or None, **native)
        u = resp.usage
        prompt = getattr(u, "prompt_tokens", 0) or 0
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return (resp.choices[0].message.content, prompt - cached,
                getattr(u, "completion_tokens", 0) or 0, cached, 0)

    async def _converse(self, messages: list[dict], params: dict) -> tuple[str | None, int, int, int, int]:
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        msgs = [{"role": m["role"], "content": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
        # Always set maxTokens explicitly: unset defaults to the model max and reserves quota.
        inference = {"maxTokens": int(params.get("max_tokens", 8192))}
        if "temperature" in params:
            inference["temperature"] = params["temperature"]
        if "top_p" in params:
            inference["topP"] = params["top_p"]
        req = {"modelId": self.model, "messages": msgs, "inferenceConfig": inference}
        if system:
            req["system"] = system
        if params.get("additional_fields"):  # e.g. {"thinking": {"type": "adaptive"}} for Claude
            req["additionalModelRequestFields"] = params["additional_fields"]
        resp = await asyncio.to_thread(self._client.converse, **req)
        # Keep only visible text blocks (skip reasoningContent from thinking models).
        blocks = resp["output"]["message"]["content"]
        text = "\n".join(b["text"] for b in blocks if "text" in b) or None
        u = resp.get("usage", {})
        # Bedrock reports implicit/explicit prompt-cache tokens separately from inputTokens.
        return (text, u.get("inputTokens", 0), u.get("outputTokens", 0),
                u.get("cacheReadInputTokens", 0), u.get("cacheWriteInputTokens", 0))
