"""Build the olympiad benchmark from MathArena's final-answer competitions (Hugging Face).

Temporal split (newer = test), so tuning never sees test problems and contamination can be
reported by year:
  tune  2023-2024: AIME 2023 I/II, AIME 2024 I/II, HMMT Feb 2023, HMMT Feb 2024        (120)
  test  2025-2026: AIME 2025/2026, HMMT Feb 2025/2026, HMMT Nov 2025, BRUMO 2025,
                   SMT 2025, CMIMC 2025                                                 (276)

License: MathArena datasets are CC BY-NC-SA 4.0 (non-commercial, share-alike). They're downloaded
on demand and NOT committed (data/olympiad/ is gitignored). Cite MathArena when you use it.

    python -m yks.olympiad
"""
from __future__ import annotations

import ast
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "olympiad"
ROWS = "https://datasets-server.huggingface.co/rows?dataset=MathArena/{ds}&config=default&split=train&offset={off}&length=100"

TUNE = ["aime_2023_I", "aime_2023_II", "aime_2024_I", "aime_2024_II", "hmmt_feb_2023", "hmmt_feb_2024"]
TEST = ["aime_2025", "aime_2026", "hmmt_feb_2025", "hmmt_feb_2026", "hmmt_nov_2025", "brumo_2025",
        "smt_2025", "cmimc_2025"]
FIGURE_RE = re.compile(r"\[asy\]|figure|diagram|shown below|pictured", re.IGNORECASE)


def _get(url: str, tries: int = 6) -> dict:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == tries - 1:
                raise
            time.sleep(2 ** attempt * 3)  # HF datasets-server rate limit: back off
    raise RuntimeError("unreachable")


def fetch(ds: str) -> list[dict]:
    cache = ROOT / "data" / "raw" / "olympiad" / f"{ds}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows, off = [], 0
    while True:
        page = _get(ROWS.format(ds=ds, off=off))
        rows += [x["row"] for x in page["rows"]]
        off += 100
        if off >= page.get("num_rows_total", 0):
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    time.sleep(1)
    return rows


def _topic(raw) -> str:
    """problem_type is a list (sometimes stringified) like ['Algebra']; keep the first tag."""
    if not raw:
        return "?"
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = [raw]
    return str(raw[0]) if raw else "?"


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, comps in (("tune", TUNE), ("test", TEST)):
        items = []
        for ds in comps:
            for r in fetch(ds):
                ans = str(r["answer"]).strip()
                items.append({
                    "id": f"{ds}-{int(r['problem_idx']):02d}",
                    "subject": re.sub(r"_(I|II)$", "", ds),   # competition, e.g. aime_2025
                    "year": int(re.search(r"20\d\d", ds).group()),
                    "topic": _topic(r.get("problem_type")),
                    "question": r["problem"],
                    "answer": ans,
                    "answer_type": "integer" if ds.startswith("aime") else "expr",
                    "has_figure_ref": bool(FIGURE_RE.search(r["problem"])),
                })
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in sorted(items, key=lambda x: x["id"]):
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)} problems from {len(comps)} competitions "
              f"(figure-referencing: {sum(i['has_figure_ref'] for i in items)})")


if __name__ == "__main__":
    build()
