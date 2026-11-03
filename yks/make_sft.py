"""Build a Fireworks SFT dataset by distilling frontier models on SYNTHETIC YKS-style questions.

Why synthetic: the public TurkishMMLU STEM set is our held-out eval, so it must never be trained on.
Pipeline (each step is a quality gate):
  1. generator model writes an original 5-option question for a curriculum topic (JSON)
  2. an independent verifier model solves it with the exact eval prompt
  3. keep only if the verifier's answer == generator's key (cross-model agreement)
  4. decontaminate: drop anything that overlaps eval questions (word 8-gram or high 3-gram Jaccard)
  5. write {"messages": [system, user, assistant]} JSONL (Fireworks/OpenAI chat format)

    python -m yks.make_sft --n 600 --generator gpt-6.1-sol --verifier claude-opus-5.5
Outputs data/sft/{train,val}.jsonl and data/sft/provenance.jsonl. Prints cost so you can disclose it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from tqdm.asyncio import tqdm

from .client import Model, load_config
from .data import load
from .prompts import build_messages, extract_answer
from .topics import TOPICS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sft"


GEN_PROMPT = (
    "Türkiye YKS (TYT/AYT) formatında, {subject} dersinden '{topic}' konusunda ÖZGÜN bir çoktan seçmeli soru yaz. "
    "Zorluk: {difficulty}. Geçmiş ÖSYM sorularını veya yayınlanmış soruları kopyalama. Şekil gerektirmesin; "
    "gerekli tüm bilgi metinde olsun. Tam 5 şık olsun, tek doğru cevap olsun, çeldiriciler öğrencilerin tipik "
    "hatalarından gelsin. Sadece JSON döndür: "
    '{{"question": "...", "choices": ["...","...","...","...","..."], "answer": "A-E"}}'
)


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


class Decontaminator:
    def __init__(self, texts: list[str]):
        self.g8 = set().union(*(_ngrams(t, 8) for t in texts))
        self.g3 = [_ngrams(t, 3) for t in texts]

    def is_contaminated(self, text: str) -> bool:
        if _ngrams(text, 8) & self.g8:
            return True
        q3 = _ngrams(text, 3)
        return any(q3 and g and len(q3 & g) / len(q3 | g) > 0.5 for g in self.g3)


def _parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not d or len(d.get("choices", [])) != 5 or d.get("answer") not in list("ABCDE"):
        return None
    return d


async def make_one(gen: Model, ver: Model, subject: str, topic: str, difficulty: str) -> dict:
    g = await gen.chat([{"role": "user", "content": GEN_PROMPT.format(subject=subject, topic=topic,
                                                                         difficulty=difficulty)}])
    cost = g.cost_usd
    q = _parse_json(g.text)
    if not q:
        return {"status": "bad_json", "cost": cost}
    item = {"question": q["question"], "choices": [str(c) for c in q["choices"]], "answer": q["answer"]}
    v = await ver.chat(build_messages(item))
    cost += v.cost_usd
    pred = extract_answer(v.text)
    if pred != item["answer"]:
        return {"status": "disagree", "cost": cost}
    solution = v.text.rsplit("</think>", 1)[-1].strip()
    # Train "reason first, answer last": the target must END with the answer line, like the eval prompt asks.
    last_line = solution.splitlines()[-1] if solution else ""
    if extract_answer(last_line) != pred:
        solution = f"{solution}\n\nCevap: {pred}"
    return {"status": "ok", "cost": cost, "subject": subject, "topic": topic, "difficulty": difficulty,
            "item": item, "solution": solution}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="generation attempts (expect ~60-80%% to survive)")
    ap.add_argument("--generator", default="gpt-6.1-sol")
    ap.add_argument("--verifier", default="claude-opus-5.5")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    cfg = load_config()
    gen, ver = Model.from_config(cfg, a.generator), Model.from_config(cfg, a.verifier)
    eval_texts = [i["question"] + " " + " ".join(i["choices"]) for s in ("test", "tune") for i in load(s)]
    decon = Decontaminator(eval_texts)

    rng = random.Random(a.seed)
    jobs = []
    for _ in range(a.n):
        subject = rng.choice(list(TOPICS))
        jobs.append((subject, rng.choice(TOPICS[subject]), rng.choice(["kolay", "orta", "zor", "zor"])))

    sem = asyncio.Semaphore(a.concurrency)

    async def run(job):
        async with sem:
            return await make_one(gen, ver, *job)

    results = await tqdm.gather(*[run(j) for j in jobs], desc="distill")
    total_cost = sum(r["cost"] for r in results)
    kept, seen, stats = [], set(), {"bad_json": 0, "disagree": 0, "contaminated": 0, "duplicate": 0}
    for r in results:
        if r["status"] != "ok":
            stats[r["status"]] += 1
            continue
        text = r["item"]["question"] + " " + " ".join(r["item"]["choices"])
        if decon.is_contaminated(text):
            stats["contaminated"] += 1
            continue
        key = re.sub(r"\W+", "", r["item"]["question"].lower())[:200]
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(r)

    rng.shuffle(kept)
    n_val = max(1, round(len(kept) * a.val_frac)) if kept else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("val", kept[:n_val]), ("train", kept[n_val:])):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                msgs = build_messages(r["item"]) + [{"role": "assistant", "content": r["solution"]}]
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    with open(OUT / "provenance.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({k: r[k] for k in ("subject", "topic", "difficulty", "item")}, ensure_ascii=False) + "\n")
    summary = {"attempts": a.n, "kept": len(kept), "train": len(kept) - n_val, "val": n_val, **stats,
               "generator": a.generator, "verifier": a.verifier, "est_cost_usd": round(total_cost, 2)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
