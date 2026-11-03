"""Wait for the SFT job, deploy the fine-tuned model, evaluate it (yks-sft), and ALWAYS delete the deployment.

    python -m yks.deploy_sft            # reads results/sft_job.json written by yks.submit_sft
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from .client import load_config
from .ladder import ACCOUNT, fw, route
from .run_eval import run_system

ROOT = Path(__file__).resolve().parent.parent
PRICE = {"H100": 8.00, "H200": 8.00, "B200": 13.00, "B300": 15.00}  # fireworks.ai/pricing, $/GPU-hr


def job_state(job_id: str) -> str:
    out = fw("supervised-fine-tuning-job", "get", job_id, check=False)
    states = re.findall(r"State:\s+(JOB_STATE_\w+)", out)
    return states[-1] if states else "UNKNOWN"


def shapes_for(model: str) -> list[tuple[str, int, str]]:
    out = fw("deployment-shape-version", "match", "--model", model, check=False)
    rows = []
    for line in out.splitlines()[1:]:
        m = re.match(r"(\S+) \(\S+\)\s+\S+\s+(\d+)\s+NVIDIA_(\w+?)_", line)
        if m:
            rows.append((m.group(1), int(m.group(2)), m.group(3)))
    return sorted(rows, key=lambda r: r[1] * PRICE.get(r[2], 99))  # cheapest first


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="yks-sft")
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--job-file", default="results/sft_job.json")
    a = ap.parse_args()
    job = json.loads((ROOT / a.job_file).read_text())
    model = job["output_model"]

    t_job = time.time()
    while (s := job_state(job["job_id"])) not in ("JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
        print(f"[{time.strftime('%H:%M:%S')}] {job['job_id']}: {s} ({(time.time() - t_job) / 60:.0f} min)", flush=True)
        await asyncio.sleep(45)
    print(f"job finished: {s}", flush=True)
    if s != "JOB_STATE_COMPLETED":
        raise SystemExit(fw("supervised-fine-tuning-job", "get", job["job_id"], check=False)[-1500:])

    shapes = shapes_for(model)
    print("deployable shapes:", shapes, flush=True)
    cfg = load_config()
    dep_id = f"{a.system.replace('.', 'p')}-{int(time.time()) % 100000}"
    rec = {"system": a.system, "model": model, "deployment": dep_id, "attempts": []}
    t0, created = time.time(), False
    try:
        for shape, gpus, accel in shapes or [("default", 1, "H200")]:
            try:
                fw("deployment", "create", model, "--deployment-id", dep_id, "--deployment-shape", shape,
                   "--min-replica-count", "1", "--max-replica-count", "1", "--wait", "--wait-timeout", "40m")
                created = True
                rec.update(shape=shape, gpus=gpus, accelerator=accel, gpu_hour_usd=PRICE.get(accel, 13.0))
                break
            except RuntimeError as e:
                rec["attempts"].append({"shape": shape, "error": str(e)[-300:]})
                print(f"shape {shape} failed: {str(e)[-200:]}", flush=True)
                fw("deployment", "delete", dep_id, "--ignore-checks", check=False)
        if not created:
            raise RuntimeError("could not deploy the fine-tuned model on any shape")
        rec["ready_after_s"] = round(time.time() - t0)
        mname = cfg["systems"][a.system]["model"]
        cands_model = cfg["models"][mname]
        cands_model["model"] = model
        rec["route"] = await route(cfg, mname, model, dep_id)
        cands_model.update(billing="gpu_hour", gpu_hour_usd=rec["gpu_hour_usd"], gpus=rec["gpus"])
        print(f"ready in {rec['ready_after_s']}s on {rec['shape']}, route={rec['route']}", flush=True)
        t1 = time.time()
        await run_system(a.system, a.split, None, a.concurrency, cfg)
        rec["eval_s"] = round(time.time() - t1)
    finally:
        out = fw("deployment", "delete", dep_id, "--ignore-checks", check=False) if created else "never created"
        rec["deleted_out"] = out.strip()[-200:]
        rec["total_billed_s"] = round(time.time() - t0)
        rec["est_cost_usd"] = round(rec["total_billed_s"] / 3600 * rec.get("gpus", 0) * rec.get("gpu_hour_usd", 0), 2)
        with (ROOT / "results" / "ladder_deployments.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("done:", json.dumps({k: rec.get(k) for k in ("route", "eval_s", "total_billed_s", "est_cost_usd")}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
