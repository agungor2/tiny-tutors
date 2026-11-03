"""Benchmark tutoring quality: does a small model diagnose mistakes as well as a frontier model?

For each question, a simulated student picks a wrong option (seeded, identical for every tutor). Each
tutor diagnoses it. A judge model from a DIFFERENT vendor than the tutors being compared scores each
diagnosis 1-5 on a rubric (it sees the answer key):
  correctness   solution is right and reaches the key
  misconception plausibly explains why the chosen distractor is tempting
  hint_ladder   hints escalate and none gives the answer away
Also reported: key-consistency rate (guardrail), parse rate, latency, cost per diagnosis.

    python -m yks.tutor_eval --tutors nemotron-lightning claude-opus-5.5 --judge gpt-6.1-sol --n 60
    # olympiad: the "students" are REAL wrong answers from an eval run (e.g. a small model's mistakes)
    python -m yks.tutor_eval --dataset olympiad --wrong-from qwen8 --tutors gpt-oss-120b claude-opus-5.5 --judge gpt-6.1-sol
LLM-as-judge is an approximation; spot-check ~20 by hand (the teacher tab is built for that) and say so.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path

from tqdm.asyncio import tqdm

from .client import Model, load_config
from .data import load
from .prompts import LETTERS, format_question
from .tutor import Tutor, parse_json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tutor"

JUDGE = (
    "You grade a tutor's diagnosis of a student's wrong answer on a Turkish university entrance exam "
    "(YKS) or math olympiad question. Correct answer: {gold}. Student answered: {chosen}. Score 1-5 each:\n"
    "correctness: the solution is mathematically/scientifically right and reaches {gold}.\n"
    "misconception: the stated error plausibly explains why a student would pick {chosen}.\n"
    "hint_ladder: 3 hints that escalate in specificity and do NOT reveal the final answer.\n"
    'Reply with JSON only: {{"correctness": n, "misconception": n, "hint_ladder": n}}'
)


async def judge(model: Model, item: dict, chosen: str, diag: dict) -> dict:
    body = {k: diag[k] for k in ("misconception", "why_tempting", "hints", "solution")}
    r = await model.chat([{"role": "system", "content": JUDGE.format(gold=item["answer"], chosen=chosen)},
                          {"role": "user", "content": f"{format_question(item)}\n\n--- TUTOR OUTPUT ---\n"
                                                      f"{json.dumps(body, ensure_ascii=False)}"}])
    return parse_json(r.text) or {}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutors", nargs="+", required=True)
    ap.add_argument("--judge", default="gpt-6.1-sol")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="yks", choices=["yks", "olympiad"])
    ap.add_argument("--wrong-from", help="system whose wrong answers (results/<dataset>/<split>/<sys>.jsonl) "
                    "become the student mistakes; required for olympiad, optional for yks")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    cfg = load_config()
    rng = random.Random(a.seed)
    items = {i["id"]: i for i in load(a.split, a.dataset) if not i["has_figure_ref"]}
    if a.wrong_from:
        src = ROOT / "results" / a.dataset / a.split / f"{a.wrong_from}.jsonl"
        rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        seen, cases = set(), []
        for r in rows:  # one real mistake per problem (first repeat), skip blanks/unparsed
            pid = r["id"].split("#")[0]
            if pid in items and pid not in seen and not r["correct"] and r["pred"] not in (None, "BOŞ"):
                seen.add(pid)
                cases.append((items[pid], r["pred"]))
        rng.shuffle(cases)
        cases = cases[:a.n]
    elif a.dataset == "yks":
        pool = list(items.values())
        rng.shuffle(pool)
        cases = [(i, rng.choice([L for L in LETTERS if L != i["answer"]])) for i in pool[:a.n]]
    else:
        raise SystemExit("olympiad needs --wrong-from <system>: free-response mistakes can't be simulated by picking a letter")
    if not cases:
        raise SystemExit("no wrong answers to diagnose")
    jm = Model.from_config(cfg, a.judge)
    sem = asyncio.Semaphore(a.concurrency)
    OUT.mkdir(parents=True, exist_ok=True)
    kind = f"real wrong answers from `{a.wrong_from}`" if a.wrong_from else "simulated wrong answers"
    lines = [f"# Tutor quality ({a.dataset}, n={len(cases)} {kind}, {a.split} split, judge `{a.judge}`)", "",
             "| tutor | correctness | misconception | hint ladder | key-consistent | parsed | p50 latency s | $/diagnosis |",
             "|---|---|---|---|---|---|---|---|"]
    for name in a.tutors:
        tutor = Tutor(name, cfg, a.dataset)

        async def one(case):
            item, chosen = case
            async with sem:
                d = await tutor.diagnose(item, chosen)
                s = await judge(jm, item, chosen, d) if d["parsed"] else {}
            return {"id": item["id"], "chosen": chosen, "gold": item["answer"], "scores": s,
                    **{k: d[k] for k in ("topic", "misconception", "why_tempting", "hints", "solution",
                                         "consistent", "parsed", "latency_s", "cost_usd", "errors")}}

        rows = await tqdm.gather(*[one(c) for c in cases], desc=name)
        with open(OUT / f"{a.dataset}-{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def avg(key):  # unparsed / unjudged diagnoses score 1 (worst), so failures aren't hidden
            return statistics.mean(float(r["scores"].get(key, 1) or 1) for r in rows)
        lines.append(
            f"| {name} | {avg('correctness'):.2f} | {avg('misconception'):.2f} | {avg('hint_ladder'):.2f} | "
            f"{sum(r['consistent'] for r in rows) / len(rows):.0%} | {sum(r['parsed'] for r in rows) / len(rows):.0%} | "
            f"{statistics.median(r['latency_s'] for r in rows):.1f} | "
            f"${statistics.mean(r['cost_usd'] for r in rows):.5f} |")
    (OUT / f"report-{a.dataset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
