"""Discover real model IDs so configs/systems.yaml stays honest.

    python -m yks.list_models fireworks [filter ...]   # serverless-callable Fireworks models
    python -m yks.list_models tunable [filter ...]     # Fireworks model library: fine-tunable base models
    python -m yks.list_models bedrock [filter ...]     # Bedrock OpenAI/Anthropic models + inference profiles
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

from . import client  # noqa: F401  (loads .env)
from .client import load_config


def _match(text: str, terms: list[str]) -> bool:
    return not terms or any(t in text.lower() for t in terms)


def fireworks(terms: list[str]) -> None:
    from openai import OpenAI
    c = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])
    for m in sorted(c.models.list(), key=lambda m: m.id):
        if _match(m.id, terms):
            print(m.id)


def tunable(terms: list[str]) -> None:
    """Page through the public Fireworks model library and keep models managed training supports."""
    key = os.environ["FIREWORKS_API_KEY"]
    rows, token = [], ""
    while True:
        qs = urllib.parse.urlencode({"pageSize": 200, **({"pageToken": token} if token else {})})
        req = urllib.request.Request(f"https://api.fireworks.ai/v1/accounts/fireworks/models?{qs}",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        for m in page.get("models", []):
            bd = m.get("baseModelDetails") or {}
            sft = m.get("supervisedLoraTunable") or m.get("supervisedFullParameterTunable") or m.get("tunable")
            rl = m.get("rlLoraTunable") or m.get("rlFullParameterTunable") or m.get("rlTunable")
            if (sft or rl) and _match(m["name"], terms):
                rows.append((m["name"], bd.get("parameterCount") or "?", "moe" if bd.get("moe") else "dense",
                             "SFT" if sft else "-", "RL" if rl else "-",
                             "serverless" if m.get("supportsServerless") else "-",
                             m.get("trainingContextLength") or m.get("contextLength") or "?"))
        token = page.get("nextPageToken")
        if not token:
            break

    def size(r):
        try:
            return int(r[1])
        except ValueError:
            return 1 << 62
    print(f"{'model':70} {'params':>15} {'arch':5} {'sft':3} {'rl':2} {'base serving':12} ctx")
    for r in sorted(rows, key=size):
        print(f"{r[0]:70} {r[1]:>15} {r[2]:5} {r[3]:3} {r[4]:2} {r[5]:12} {r[6]}")
    print(f"\n{len(rows)} tunable models. Trained models are served on dedicated deployments (billing: gpu_hour).")


def bedrock(terms: list[str]) -> None:
    import boto3
    region = load_config()["providers"]["bedrock"].get("region", "us-east-1")
    b = boto3.client("bedrock", region_name=region)
    terms = terms or ["openai", "anthropic"]
    print(f"# foundation models ({region})")
    for m in b.list_foundation_models()["modelSummaries"]:
        if _match(m["modelId"], terms):
            print(f"{m['modelId']:55} {','.join(m.get('inferenceTypesSupported', []))}")
    print("\n# inference profiles (use these IDs when on-demand base IDs are rejected)")
    paginator = b.get_paginator("list_inference_profiles")
    for page in paginator.paginate():
        for p in page["inferenceProfileSummaries"]:
            if _match(p["inferenceProfileId"], terms):
                print(p["inferenceProfileId"])


if __name__ == "__main__":
    mode, *rest = sys.argv[1:] or ["fireworks"]
    terms = [t.lower() for t in rest]
    {"fireworks": fireworks, "tunable": tunable, "bedrock": bedrock}[mode](terms)
