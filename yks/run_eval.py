"""Run one or more systems over a split. Results are appended per question (resumable).

    python -m yks.run_eval --systems claude-opus-5.5 nemotron-lightning --split test                 # YKS
    python -m yks.run_eval --dataset olympiad --systems claude-opus-5.5 oss120 --repeats 4
    python -m yks.run_eval --systems nemo-sc5 --split tune --limit 20                  # quick knob check

--repeats R runs each problem R times (ids get a #r suffix) to average out sampling noise on small
olympiad sets; every system must use the same R to be compared.
Results: results/<dataset>/<split>/<system>.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from tqdm.asyncio import tqdm

from .client import load_config
from .data import DATASETS, load
from .strategies import System

ROOT = Path(__file__).resolve().parent.parent


async def run_system(name: str, split: str, limit: int | None, concurrency: int, cfg: dict,
                     dataset: str = "yks", repeats: int = 1, overrides: dict | None = None) -> Path:
    system = System(name, cfg, overrides)
    base = load(split, dataset)[:limit] if limit else load(split, dataset)
    items = base if repeats == 1 else [{**it, "id": f"{it['id']}#{r}"} for it in base for r in range(repeats)]
    out_dir = ROOT / "results" / dataset / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.jsonl"
    done = set()
    if out.exists():
        # Re-run only rows that failed purely due to API errors (report keeps the last row per id).
        rows = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
        done = {r["id"] for r in rows if not (r["pred"] is None and r["errors"])}
    todo = [it for it in items if it["id"] not in done]
    (out_dir / f"{name}.config.json").write_text(json.dumps(
        {**system.describe(), "models": cfg["models"], "dataset": dataset, "split": split, "repeats": repeats,
         "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2, ensure_ascii=False))

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(item: dict) -> None:
        async with sem:
            rec = await system.predict(item)
        row = {"id": item["id"], "subject": item["subject"], "topic": item.get("topic"),
               "year": item.get("year"), "gold": item["answer"], "has_figure_ref": item["has_figure_ref"], **rec}
        async with lock:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[{name}] {len(todo)} to run ({len(done)} cached) -> {out}")
    t0 = time.perf_counter()
    await tqdm.gather(*[one(it) for it in todo], desc=name)
    # Wall time per invocation: used to charge GPU-hours for dedicated (gpu_hour) deployments.
    if todo:
        with (out_dir / f"{name}.runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"n": len(todo), "wall_s": time.perf_counter() - t0,
                                "concurrency": concurrency}) + "\n")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--dataset", default="yks", choices=DATASETS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, help="override max_tokens for every call (olympiad: 32000+)")
    ap.add_argument("--split", default="test", choices=["tune", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Keep equal across systems when comparing latency; note it in your writeup.")
    a = ap.parse_args()
    cfg = load_config()
    for name in a.systems:  # sequential so systems don't compete for rate limits / skew latency
        await run_system(name, a.split, a.limit, a.concurrency, cfg, a.dataset, a.repeats,
                         {"max_tokens": a.max_tokens} if a.max_tokens else None)


if __name__ == "__main__":
    asyncio.run(main())
