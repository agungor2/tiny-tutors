"""Benchmark Fireworks-tunable base models that aren't on serverless, one dedicated deployment at a time.

For each model: create a deployment on its MINIMAL shape -> wait until READY -> find the routing
string -> run yks.run_eval at high concurrency (realistic GPU utilisation) -> ALWAYS delete the
deployment (try/finally), so nothing keeps billing.

    python -m yks.ladder --split test --systems qwen3-0.6b qwen3-4b qwen3-8b qwen3-14b

Cost accounting: billing gpu_hour at --gpu-hour-usd (Fireworks list price for the shape's GPU) x
measured eval wall time. Deployment spin-up time is excluded (one-off, not per-question); it IS
billed, so it's printed for disclosure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from .client import Model, load_config
from .run_eval import run_system

ROOT = Path(__file__).resolve().parent.parent
FW = str(ROOT / ".tools" / "fw")  # firectl wrapper that reads the key from .env
ACCOUNT = os.environ.get("FIREWORKS_ACCOUNT_ID", "YOUR_FIREWORKS_ACCOUNT")

# system -> candidate (shape, gpus, accelerator, $/GPU-hr) from `firectl deployment-shape-version match`,
# cheapest first. Some shape versions get blocked server-side, so we fall through to the next one.
S = "accounts/fireworks/deploymentShapes/"
H200, B200, B300 = 8.00, 13.00, 15.00  # fireworks.ai/pricing (2026-11-02)
SHAPES = {
    "qwen3-0.6b": [(S + "qwen3-0p6b-minimal", 1, "H200", H200), (S + "rft-qwen3-0p6b", 1, "B200", B200)],
    "qwen3-4b": [(S + "qwen3-4b-minimal", 1, "H200", H200), (S + "rft-qwen3-4b", 1, "B200", B200)],
    "qwen3-8b": [(S + "qwen3-8b-minimal", 1, "H200", H200), (S + "rft-qwen3-8b", 1, "B200", B200),
                 (S + "qwen3-8b-b300-fp8", 1, "B300", B300)],
    "qwen3-14b": [(S + "qwen3-14b-minimal", 1, "H200", H200), (S + "rft-qwen3-14b", 2, "H200", H200)],
}
LOG = ROOT / "results" / "ladder_deployments.jsonl"


def fw(*args: str, check: bool = True) -> str:
    p = subprocess.run([FW, *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"firectl {' '.join(args[:3])} failed: {p.stderr.strip()[-600:]}")
    return p.stdout + p.stderr


async def route(cfg: dict, name: str, base_model: str, dep_id: str) -> str:
    """Find the model string that routes to our deployment (format differs by deployment type)."""
    candidates = [f"{base_model}#accounts/{ACCOUNT}/deployments/{dep_id}",
                  f"accounts/{ACCOUNT}/deployments/{dep_id}", base_model]
    last = ""
    for _ in range(30):  # READY can precede routability by a little
        for c in candidates:
            cfg["models"][name]["model"] = c
            r = await Model.from_config(cfg, name).chat([{"role": "user", "content": "Say OK."}], max_tokens=8)
            if not r.error:
                return c
            last = r.error
        await asyncio.sleep(20)
    raise RuntimeError(f"deployment {dep_id} never became routable: {last}")


async def run_one(name: str, split: str, concurrency: int, dataset: str) -> None:
    cfg = load_config()
    base_model = cfg["models"][cfg["systems"][name]["model"]]["model"]
    dep_id = f"yks-{name.replace('.', 'p')}-{int(time.time()) % 100000}"
    t0 = time.time()
    rec = {"system": name, "deployment": dep_id, "created": time.strftime("%H:%M:%S"), "attempts": []}
    created = False
    try:
        for shape, gpus, accel, gpu_hour_usd in SHAPES[name]:
            print(f"[{name}] creating {dep_id} on {shape} ...", flush=True)
            try:
                fw("deployment", "create", base_model, "--deployment-id", dep_id, "--deployment-shape", shape,
                   "--min-replica-count", "1", "--max-replica-count", "1", "--wait", "--wait-timeout", "40m")
                created = True
                rec.update(shape=shape, gpus=gpus, accelerator=accel, gpu_hour_usd=gpu_hour_usd)
                break
            except RuntimeError as e:
                rec["attempts"].append({"shape": shape, "error": str(e)[-300:]})
                print(f"[{name}] shape failed: {str(e)[-200:]}", flush=True)
                fw("deployment", "delete", dep_id, "--ignore-checks", check=False)  # clean any partial
        if not created:
            raise RuntimeError(f"no deployable shape for {name}")
        rec["ready_after_s"] = round(time.time() - t0)
        model_name = cfg["systems"][name]["model"]
        rec["route"] = await route(cfg, model_name, base_model, dep_id)
        cfg["models"][model_name].update(billing="gpu_hour", gpu_hour_usd=rec["gpu_hour_usd"], gpus=rec["gpus"])
        print(f"[{name}] ready in {rec['ready_after_s']}s, route={rec['route']}", flush=True)
        t1 = time.time()
        # Olympiad problems need a long reasoning budget (same 32K cap the other olympiad systems used).
        await run_system(name, split, None, concurrency, cfg, dataset,
                         overrides={"max_tokens": 32000} if dataset == "olympiad" else None)
        rec["eval_s"] = round(time.time() - t1)
    finally:
        out = fw("deployment", "delete", dep_id, "--ignore-checks", check=False) if created else "never created"
        rec["deleted"] = (not created) or "error" not in out.lower() or "notfound" in out.lower().replace(" ", "")
        rec["deleted_out"] = out.strip()[-200:]
        rec["total_billed_s"] = round(time.time() - t0)
        rec["est_cost_usd"] = round(rec["total_billed_s"] / 3600 * rec.get("gpus", 0) * rec.get("gpu_hour_usd", 0), 2)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{name}] deleted={rec['deleted']} billed~{rec['total_billed_s']}s (~${rec['est_cost_usd']})", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SHAPES))
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="yks")
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    for name in a.systems:  # strictly one deployment at a time; one failure doesn't stop the ladder
        try:
            await run_one(name, a.split, a.concurrency, a.dataset)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] SKIPPED: {str(e)[-300:]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
