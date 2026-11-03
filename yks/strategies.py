"""Inference strategies. Each returns one prediction record per question.

single            one call
self_consistency  k parallel samples, majority vote (latency = slowest sample)
cascade           small model votes k times; escalate to large model only if votes disagree
"""
from __future__ import annotations

import asyncio

from .client import CallResult, Model, load_config
from .grading import extract, is_correct, vote
from .prompts import BLANK, build_messages


def _agg(calls: list[CallResult], parallel: bool) -> dict:
    lat = [c.latency_s for c in calls]
    return {
        "latency_s": max(lat) if parallel else sum(lat),
        "in_tokens": sum(c.in_tokens for c in calls),
        "out_tokens": sum(c.out_tokens for c in calls),
        "cost_usd": sum(c.cost_usd for c in calls),
        "errors": [c.error for c in calls if c.error],
    }


async def _sample(model: Model, item: dict, k: int, params: dict, mode: str = "standard") -> list[CallResult]:
    msgs = build_messages(item, mode)
    return await asyncio.gather(*[model.chat(msgs, **params) for _ in range(k)])


class System:
    def __init__(self, name: str, cfg: dict | None = None, overrides: dict | None = None):
        cfg = cfg or load_config()
        self.name = name
        # Run-level params applied to every call (e.g. a larger max_tokens for olympiad problems).
        self.overrides = dict(overrides or {})
        self.spec = cfg["systems"][name]
        self.strategy = self.spec["strategy"]
        if self.strategy == "cascade":
            self.small = Model.from_config(cfg, self.spec["small"])
            self.large = Model.from_config(cfg, self.spec["large"])
        else:
            self.model = Model.from_config(cfg, self.spec["model"])

    def describe(self) -> dict:
        return {"system": self.name, **self.spec, "run_overrides": self.overrides}

    async def predict(self, item: dict) -> dict:
        s = self.spec
        mode = s.get("prompt", "standard")  # "exam" = YKS scoring with BOŞ allowed
        if self.strategy in ("single", "self_consistency"):
            k = s.get("k", 1)
            calls = await _sample(self.model, item, k, {**self.overrides, **(s.get("params") or {})}, mode)
            preds = [extract(item, c.text) for c in calls]
            pred, agreement = vote(item, preds)
            rec = {"pred": pred, "votes": preds, "agreement": agreement, "escalated": False,
                   **_agg(calls, parallel=True), "sample_text": calls[0].text}
        elif self.strategy == "cascade":
            small = await _sample(self.small, item, s.get("k", 3), {**self.overrides, **(s.get("small_params") or {})}, mode)
            preds = [extract(item, c.text) for c in small]
            pred, agreement = vote(item, preds)
            stage1 = _agg(small, parallel=True)
            escalate = agreement < s.get("escalate_if_agreement_below", 1.0)
            text = small[0].text
            if escalate:
                big = await self.large.chat(build_messages(item, mode), **{**self.overrides, **(s.get("large_params") or {})})
                big_pred = extract(item, big.text)
                pred = big_pred or pred
                text = big.text
                stage2 = _agg([big], parallel=True)
                for key in ("in_tokens", "out_tokens", "cost_usd"):
                    stage1[key] += stage2[key]
                stage1["latency_s"] += stage2["latency_s"]  # sequential
                stage1["errors"] += stage2["errors"]
            rec = {"pred": pred, "votes": preds, "agreement": agreement, "escalated": escalate,
                   **stage1, "sample_text": text}
        else:
            raise ValueError(f"unknown strategy {self.strategy}")
        rec["correct"] = is_correct(item, rec["pred"])
        rec["parse_fail"] = rec["pred"] is None
        rec["abstained"] = rec["pred"] == BLANK
        return rec
