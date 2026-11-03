"""Wait for yks.make_sft to finish, then upload the datasets and start a Fireworks managed SFT job.

    python -m yks.submit_sft --base nemotron-lightning-3p5-30b-a3b --output yks-sft-nemotron

Writes results/sft_job.json with the job id so later steps (deploy + eval) can pick it up.
Check progress: .tools/fw supervised-fine-tuning-job get <job-id>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FW = str(ROOT / ".tools" / "fw")
SFT = ROOT / "data" / "sft"


def fw(*args: str) -> str:
    p = subprocess.run([FW, *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"firectl {' '.join(args[:2])} failed: {p.stderr.strip()[-800:]}")
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="nemotron-lightning-3p5-30b-a3b")
    ap.add_argument("--output", default="yks-sft-nemotron")
    ap.add_argument("--max-context", default="16384")
    ap.add_argument("--min-train", type=int, default=200, help="refuse to train on fewer rows")
    a = ap.parse_args()

    summary = SFT / "summary.json"
    while not summary.exists():  # make_sft writes this last
        time.sleep(20)
    s = json.loads(summary.read_text())
    print("data:", s, flush=True)
    if s["train"] < a.min_train:
        raise SystemExit(f"only {s['train']} training rows (< {a.min_train}); not submitting")

    stamp = time.strftime("%H%M")
    train_ds, val_ds = f"yks-sft-train-{stamp}", f"yks-sft-val-{stamp}"
    print(fw("dataset", "create", train_ds, str(SFT / "train.jsonl")), flush=True)
    print(fw("dataset", "create", val_ds, str(SFT / "val.jsonl")), flush=True)
    job_id = f"yks-sft-{stamp}"
    out = fw("supervised-fine-tuning-job", "create",
             "--base-model", f"accounts/fireworks/models/{a.base}",
             "--dataset", train_ds, "--evaluation-dataset", val_ds,
             "--output-model", a.output, "--job-id", job_id,
             "--max-context-length", a.max_context,
             "--display-name", "YKS SFT (synthetic, cross-verified, decontaminated)")
    print(out, flush=True)
    rec = {"job_id": job_id, "base": a.base, "output_model": f"accounts/YOUR_FIREWORKS_ACCOUNT/models/{a.output}",
           "train_dataset": train_ds, "val_dataset": val_ds, "data": s, "submitted": time.strftime("%H:%M:%S")}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "sft_job.json").write_text(json.dumps(rec, indent=2))
    print("submitted:", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
