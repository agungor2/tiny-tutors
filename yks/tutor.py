"""Tutor engine: diagnose a student's wrong answer, build a hint ladder, and generate verified practice.

All calls go through the same Model client as the eval, so cost and latency are measured identically.

    python -m yks.tutor build-pool --model nemotron-lightning      # topic-tag the practice pool -> data/pool.jsonl
    python -m yks.tutor diagnose --id chem-000 --student A --model nemotron-lightning
    python -m yks.tutor practice --topic "Mathematics / olasılık" --model nemotron-lightning
    python -m yks.tutor --dataset olympiad diagnose --id aime_2025-03 --student 42 --model gpt-oss-120b
    python -m yks.tutor --dataset olympiad practice --topic "Number Theory / modular arithmetic"
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
from pathlib import Path

from .client import Model, load_config
from .data import load
from .grading import extract, is_correct
from .prompts import build_messages, format_question
from .topics import topics_for

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "pool.jsonl"

DIAGNOSE_PROMPT = (
    "Sen YKS'ye hazırlanan öğrencilere koçluk yapan deneyimli bir öğretmensin. Öğrenci aşağıdaki soruda "
    "YANLIŞ şıkkı işaretledi. Doğru cevap ve öğrencinin cevabı verildi. Türkçe, kısa ve net yaz.\n"
    "Konu etiketi şu listeden TAM OLARAK biri olmalı:\n{topics}\n\n"
    "Sadece şu JSON'u döndür:\n"
    '{{"topic": "<listeden biri>", '
    '"misconception": "<öğrencinin yaptığı olası hata, 3-8 kelimelik etiket>", '
    '"why_tempting": "<öğrencinin seçtiği şıkka hangi hatalı düşünce götürür, 1-2 cümle>", '
    '"hints": ["<ipucu 1: cevabı vermeden yön göster>", "<ipucu 2: daha somut, ilk adımı söyle>", '
    '"<ipucu 3: neredeyse çözüm, son adımı öğrenciye bırak>"], '
    '"solution": "<adım adım kısa çözüm>", "solution_answer": "<A-E>"}}'
)

PRACTICE_PROMPT = (
    "YKS (TYT/AYT) formatında '{topic}' konusunda ÖZGÜN bir çoktan seçmeli soru yaz. Zorluk: {difficulty}. "
    "Bu soru, şu hatayı yapan bir öğrenciye pratik olsun: '{misconception}'. Şıklardan biri tam olarak bu "
    "hatayı yapan öğrencinin bulacağı sonuç olsun. Şekil gerektirmesin, tek doğru cevap olsun, 5 şık olsun. "
    "Sadece JSON döndür: "
    '{{"question": "...", "choices": ["...","...","...","...","..."], "answer": "<A-E>", '
    '"trap_option": "<hatayı yapan öğrencinin seçeceği şık, A-E>"}}'
)

# Olympiad (free-response, English). The "student" gave a final answer, not an option letter.
OLY_DIAGNOSE_PROMPT = (
    "You are an experienced math olympiad coach. A student gave a WRONG final answer to the problem below. "
    "You are given the correct answer and the student's answer. Be concise.\n"
    "The topic label must be EXACTLY one of:\n{topics}\n\n"
    "Return only this JSON:\n"
    '{{"topic": "<one label from the list>", '
    '"misconception": "<the likely error, 3-8 word label>", '
    '"why_tempting": "<which flawed step most plausibly produces the student\'s answer, 1-2 sentences>", '
    '"hints": ["<hint 1: a direction, no answer>", "<hint 2: the key idea or first step>", '
    '"<hint 3: nearly the solution, leave the last step>"], '
    '"solution": "<concise step-by-step solution>", "solution_answer": "<final answer, LaTeX>"}}'
)

OLY_PRACTICE_PROMPT = (
    "Write an ORIGINAL competition problem (AIME/HMMT style, difficulty: {difficulty}) on '{topic}' for a "
    "student who makes this mistake: '{misconception}'. It must have a single exact final answer, need no "
    "figure, and not copy a known contest problem. Return only JSON: "
    '{{"question": "...", "answer": "<exact final answer, LaTeX>", '
    '"trap_answer": "<the answer a student making that mistake would get>"}}'
)

# English variants (questions stay in Turkish; the tutor explains in English).
EN_DIAGNOSE_PROMPT = (
    "You are an experienced tutor coaching a student for Turkey's university entrance exam (YKS). The question "
    "below is in Turkish. The student chose a WRONG option. You are given the correct answer and the student's "
    "choice. Write ALL text fields in clear, concise ENGLISH.\n"
    "The topic label must be EXACTLY one of:\n{topics}\n\n"
    "Return only this JSON:\n"
    '{{"topic": "<one label from the list>", '
    '"misconception": "<the likely mistake, 3-8 word label>", '
    '"why_tempting": "<what flawed reasoning leads to the option the student chose, 1-2 sentences>", '
    '"hints": ["<hint 1: a direction, no answer>", "<hint 2: more concrete, the first step>", '
    '"<hint 3: almost the solution, leave the last step>"], '
    '"solution": "<concise step-by-step solution>", "solution_answer": "<A-E>"}}'
)

EN_PRACTICE_PROMPT = (
    "Write an ORIGINAL multiple-choice question in ENGLISH in the style of Turkey's university entrance exam "
    "(YKS TYT/AYT) on the topic '{topic}'. Difficulty: {difficulty}. It is practice for a student who makes "
    "this mistake: '{misconception}'. One option must be exactly the result a student making that mistake "
    "would get. No figures, a single correct answer, exactly 5 options. Return only JSON: "
    '{{"question": "...", "choices": ["...","...","...","...","..."], "answer": "<A-E>", '
    '"trap_option": "<the option a student making the mistake would choose, A-E>"}}'
)

TRANSLATE_PROMPT = (
    "Translate this Turkish exam question and its options into natural English. Keep all math, numbers, "
    "formulas and option letters exactly as they are. Do not solve it. Output only the translation."
)

TAG_PROMPT = (
    "Aşağıdaki YKS sorusunun konusunu şu listeden TAM OLARAK bir etiketle belirt:\n{topics}\n\n"
    'Sadece JSON döndür: {{"topic": "<listeden biri>"}}'
)


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = text.rsplit("</think>", 1)[-1]
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def snap_topic(topic: str | None, dataset: str = "yks") -> str:
    """Map a free-text topic onto the shared taxonomy (models sometimes paraphrase the label)."""
    topics = topics_for(dataset)
    if topic in topics:
        return topic
    best = difflib.get_close_matches(topic or "", topics, n=1, cutoff=0.0)
    return best[0] if best else "?"


def _meta(*calls) -> dict:
    return {"latency_s": round(sum(c.latency_s for c in calls), 2),
            "cost_usd": sum(c.cost_usd for c in calls),
            "in_tokens": sum(c.in_tokens for c in calls), "out_tokens": sum(c.out_tokens for c in calls),
            "errors": [c.error for c in calls if c.error]}


class Tutor:
    def __init__(self, model_name: str, cfg: dict | None = None, dataset: str = "yks", lang: str = "tr"):
        cfg = cfg or load_config()
        self.model = Model.from_config(cfg, model_name)
        self.name = model_name
        self.dataset = dataset
        self.lang = lang  # "en": explain in English (YKS questions themselves stay Turkish)

    def diagnose_messages(self, item: dict, student: str) -> list[dict]:
        topics = "\n".join(f"- {t}" for t in topics_for(self.dataset))
        if self.dataset != "yks":
            return [{"role": "system", "content": OLY_DIAGNOSE_PROMPT.format(topics=topics)},
                    {"role": "user", "content": f"{format_question(item)}\n\nCorrect answer: {item['answer']}\n"
                                                f"Student's answer: {student}"}]
        if self.lang == "en":
            return [{"role": "system", "content": EN_DIAGNOSE_PROMPT.format(topics=topics)},
                    {"role": "user", "content": f"{format_question(item)}\n\nCorrect answer: {item['answer']}\n"
                                                f"Student's answer: {student}"}]
        return [{"role": "system", "content": DIAGNOSE_PROMPT.format(topics=topics)},
                {"role": "user", "content": f"{format_question(item)}\n\nDoğru cevap: {item['answer']}\n"
                                            f"Öğrencinin cevabı: {student}"}]

    async def diagnose(self, item: dict, student: str) -> dict:
        msgs = self.diagnose_messages(item, student)
        r = await self.model.chat(msgs)
        d = parse_json(r.text) or {}
        hints = [h for h in (d.get("hints") or []) if isinstance(h, str)][:3]
        return {
            "topic": snap_topic(d.get("topic"), self.dataset),
            "misconception": d.get("misconception") or "?",
            "why_tempting": d.get("why_tempting") or "",
            "hints": hints,
            "solution": d.get("solution") or (r.text or ""),
            # Guardrail: the tutor's own solution must reach the answer key, otherwise flag it.
            "consistent": (is_correct(item, str(d.get("solution_answer", "")).strip()) if self.dataset != "yks"
                           else str(d.get("solution_answer", "")).strip().upper()[:1] == item["answer"]),
            "parsed": bool(d),
            "model": self.name, "prompt_messages": msgs, "raw": r.text, **_meta(r),
        }

    async def practice(self, topic: str, misconception: str, difficulty: str = "orta",
                       attempts: int = 2) -> dict | None:
        """Generate a practice question aimed at a misconception; keep it only if an independent
        solve by the same model reaches the generated key (self-verification)."""
        calls, oly = [], self.dataset != "yks"
        for _ in range(attempts):
            tmpl = OLY_PRACTICE_PROMPT if oly else (EN_PRACTICE_PROMPT if self.lang == "en" else PRACTICE_PROMPT)
            g = await self.model.chat([{"role": "user", "content": tmpl.format(
                topic=topic, misconception=misconception, difficulty=difficulty)}])
            calls.append(g)
            q = parse_json(g.text)
            if not q or not q.get("question") or not q.get("answer"):
                continue
            if not oly and (len(q.get("choices", [])) != 5 or q["answer"] not in list("ABCDE")):
                continue
            item = {"id": f"gen-{abs(hash(q['question'])) % 10**8:08d}", "subject": topic.split(" / ")[0],
                    "topic": topic, "question": q["question"], "answer": str(q["answer"]),
                    "targets_misconception": misconception, "source": "generated", "has_figure_ref": False}
            if oly:
                item.update(answer_type="expr", trap_answer=q.get("trap_answer"))
            else:
                item.update(choices=[str(c) for c in q["choices"]], trap_option=q.get("trap_option"))
            v = await self.model.chat(build_messages(item))
            calls.append(v)
            if is_correct(item, extract(item, v.text)):
                return {"item": item, "verified": True, **_meta(*calls)}
        return {"item": None, "verified": False, **_meta(*calls)}

    async def translate(self, item: dict) -> dict:
        r = await self.model.chat([{"role": "system", "content": TRANSLATE_PROMPT},
                                   {"role": "user", "content": format_question(item)}], max_tokens=4000)
        text = (r.text or "").rsplit("</think>", 1)[-1].strip()
        return {"text": text or "(translation unavailable)", **_meta(r)}

    async def tag(self, item: dict) -> str:
        topics = "\n".join(f"- {t}" for t in topics_for(self.dataset))
        r = await self.model.chat([{"role": "system", "content": TAG_PROMPT.format(topics=topics)},
                                   {"role": "user", "content": format_question(item)}])
        return snap_topic((parse_json(r.text) or {}).get("topic"), self.dataset)


def load_pool() -> list[dict]:
    """Practice pool for the tutor app. Uses the tune split + verified synthetic questions only.
    The test split is excluded: teacher-reviewed feedback is exported as training data, and test
    questions must never leak into training."""
    if POOL.exists():
        return [json.loads(l) for l in POOL.open(encoding="utf-8") if l.strip()]
    return [{**i, "topic": "?", "source": "tune"} for i in load("tune")]


async def build_pool(model_name: str, concurrency: int = 8) -> None:
    tutor = Tutor(model_name)
    items = [{**i, "source": "tune"} for i in load("tune")]
    prov = ROOT / "data" / "sft" / "provenance.jsonl"
    if prov.exists():
        for n, line in enumerate(prov.open(encoding="utf-8")):
            p = json.loads(line)
            items.append({**p["item"], "id": f"syn-{n:04d}", "subject": p["subject"],
                          "topic": f"{p['subject']} / {p['topic']}", "source": "synthetic",
                          "has_figure_ref": False})
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        if it.get("topic"):
            return it
        async with sem:
            return {**it, "topic": await tutor.tag(it)}

    tagged = await asyncio.gather(*[one(i) for i in items])
    with POOL.open("w", encoding="utf-8") as f:
        for it in tagged:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"pool: {len(tagged)} questions -> {POOL}")


async def _cli() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--dataset", default="yks", choices=["yks", "olympiad"])
    b = sub.add_parser("build-pool"); b.add_argument("--model", default="nemotron-lightning")
    d = sub.add_parser("diagnose"); d.add_argument("--id", required=True); d.add_argument("--student", required=True)
    d.add_argument("--model", default="nemotron-lightning")
    p = sub.add_parser("practice"); p.add_argument("--topic", required=True)
    p.add_argument("--misconception", default="işlem hatası"); p.add_argument("--model", default="nemotron-lightning")
    a = ap.parse_args()
    if a.cmd == "build-pool":
        await build_pool(a.model)
    elif a.cmd == "diagnose":
        pool = load_pool() + load("tune") if a.dataset == "yks" else load("tune", "olympiad") + load("test", "olympiad")
        item = next(i for i in pool if i["id"] == a.id)
        out = await Tutor(a.model, dataset=a.dataset).diagnose(item, a.student)
        print(json.dumps({k: v for k, v in out.items() if k not in ("prompt_messages", "raw")},
                         ensure_ascii=False, indent=2))
    else:
        out = await Tutor(a.model, dataset=a.dataset).practice(a.topic, a.misconception)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_cli())
