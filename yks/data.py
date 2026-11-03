"""Download TurkishMMLU STEM subjects and build a deterministic tune/test split.

TurkishMMLU (Yüksel et al., EMNLP Findings 2024): curriculum-expert written questions
for the Turkish high-school / YKS curriculum, 5-option multiple choice (A-E).
Source: https://huggingface.co/datasets/AYueksel/TurkishMMLU

Usage:
    python -m yks.data --subjects Mathematics Physics Chemistry --tune-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

HF_URL = "https://huggingface.co/datasets/AYueksel/TurkishMMLU/resolve/main/test/TurkishMMLU_{subject}.json"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Questions that refer to a figure we cannot see. Flagged, not removed, so you can report both.
FIGURE_RE = re.compile(r"şekil|grafik|görsel|yukarıdaki|aşağıdaki (çizim|resim)", re.IGNORECASE)


def download(subject: str) -> list[dict]:
    raw_dir = DATA / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{subject}.json"
    if not path.exists():
        url = HF_URL.format(subject=subject.replace(" ", "%20"))
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text(encoding="utf-8"))


def build(subjects: list[str], tune_frac: float, seed: int) -> None:
    tune, test = [], []
    for subject in subjects:
        items = []
        for i, q in enumerate(download(subject)):
            if len(q["choices"]) != 5 or q["answer"] not in "ABCDE":
                continue
            items.append({
                "id": f"{subject[:4].lower()}-{i:03d}",
                "subject": subject,
                "question": q["question"],
                "choices": q["choices"],
                "answer": q["answer"],
                "has_figure_ref": bool(FIGURE_RE.search(q["question"])),
            })
        rng = random.Random(f"{seed}-{subject}")
        rng.shuffle(items)
        n_tune = round(len(items) * tune_frac)
        tune += items[:n_tune]
        test += items[n_tune:]

    DATA.mkdir(exist_ok=True)
    for name, rows in (("tune", tune), ("test", test)):
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda r: r["id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"tune={len(tune)}  test={len(test)}  (figure-referencing in test: "
          f"{sum(r['has_figure_ref'] for r in test)})  -> {DATA}")


DATASETS = ("yks", "olympiad")


def load(split: str = "test", dataset: str = "yks") -> list[dict]:
    """yks -> data/{split}.jsonl (python -m yks.data); olympiad -> data/olympiad/{split}.jsonl
    (python -m yks.olympiad)."""
    path = DATA / f"{split}.jsonl" if dataset == "yks" else DATA / dataset / f"{split}.jsonl"
    if not path.exists():
        builder = "yks.data" if dataset == "yks" else f"yks.{dataset}"
        raise SystemExit(f"{path} not found. Build it first: python -m {builder}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["Mathematics", "Physics", "Chemistry"])
    ap.add_argument("--tune-frac", type=float, default=0.2,
                    help="Fraction held out for prompt/knob tuning. Report final numbers on test only.")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build(a.subjects, a.tune_frac, a.seed)
