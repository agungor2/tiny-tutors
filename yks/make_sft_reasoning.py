"""SFT v2: same synthetic questions as v1, but targets INCLUDE the reasoning trace.

Why: v1 targets were short visible solutions with no thinking. Fine-tuning Qwen3 on them taught the
model to skip its reasoning (median output 1,700 -> ~250 tokens) and accuracy dropped. v2 keeps the
teacher's reasoning in `reasoning_content` (Fireworks SFT trains it as the assistant's thinking).

Teacher: gpt-oss-120b (Fireworks serverless; exposes reasoning_content). A row is kept only if the
teacher's final answer matches the cross-verified key (rejection sampling, up to --tries attempts).
Same questions + same train/val split as v1, so the v1 vs v2 comparison isolates the target format.

    python -m yks.make_sft_reasoning --tries 2
Outputs data/sft_v2/{train,val}.jsonl + summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from . import client  # noqa: F401  (loads .env)
from .grading import extract, is_correct

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "data" / "sft", ROOT / "data" / "sft_v2"
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-oss-120b serverless $/1M


from .prompts import format_question

_KEYS: dict[str, str] | None = None


def item_from_row(row: dict) -> tuple[list[dict], dict]:
    """Recover (prompt messages, item with the cross-verified answer key) from a v1 SFT row.
    The key comes from provenance.jsonl, matched on the exact rendered question text."""
    global _KEYS
    if _KEYS is None:
        _KEYS = {}
        for line in (SRC / "provenance.jsonl").open(encoding="utf-8"):
            it = json.loads(line)["item"]
            _KEYS[format_question(it)] = it["answer"]
    prompt = row["messages"][:2]
    return prompt, {"choices": ["x"] * 5, "answer": _KEYS.get(prompt[1]["content"])}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="accounts/fireworks/models/gpt-oss-120b")
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12000)
    a = ap.parse_args()
    c = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"],
                    max_retries=8, timeout=600)
    sem = asyncio.Semaphore(a.concurrency)
    stats = {"kept": 0, "wrong": 0, "error": 0, "in_tok": 0, "out_tok": 0}

    async def one(row: dict) -> dict | None:
        prompt, item = item_from_row(row)
        for _ in range(a.tries):
            async with sem:
                try:
                    r = await c.chat.completions.create(model=a.teacher, messages=prompt, max_tokens=a.max_tokens,
                                                        extra_body={"reasoning_effort": "medium"})
                except Exception:  # noqa: BLE001
                    stats["error"] += 1
                    continue
            stats["in_tok"] += r.usage.prompt_tokens
            stats["out_tok"] += r.usage.completion_tokens
            m = r.choices[0].message
            rc = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content")
            if rc and m.content and is_correct(item, extract(item, m.content)):
                stats["kept"] += 1
                return {"messages": prompt + [{"role": "assistant", "reasoning_content": rc.strip(),
                                               "content": m.content.strip()}]}
            stats["wrong"] += 1
        return None

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val"):
        rows = [json.loads(l) for l in (SRC / f"{split}.jsonl").open(encoding="utf-8")]
        res = await tqdm.gather(*[one(r) for r in rows], desc=split)
        kept = [x for x in res if x]
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for x in kept:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        summary[split] = {"source_rows": len(rows), "kept": len(kept)}
    summary.update(teacher=a.teacher, tries=a.tries, **stats,
                   est_cost_usd=round((stats["in_tok"] * PRICE_IN + stats["out_tok"] * PRICE_OUT) / 1e6, 2),
                   built=time.strftime("%H:%M:%S"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
