"""Student-facing feedback: the "learning" half of the product.

Given a question and the letter a student picked, the small model explains:
  1. a Socratic hint first (so the student can retry),
  2. the full solution,
  3. why the student's distractor is tempting (the likely misconception),
  4. one similar practice question.

    python -m yks.feedback --id math-000 --student B --model nemotron-lightning
    python -m yks.feedback --id math-000 --student B --model nemotron-lightning --judge claude-opus-5.5

--judge scores the feedback with another model on a 1-5 rubric (use it on a sample to
measure feedback quality; disclose that LLM-as-judge is an approximation).
"""
from __future__ import annotations

import argparse
import asyncio
import json

from .client import Model, load_config
from .data import load
from .prompts import LETTERS

TUTOR_PROMPT = (
    "Sen YKS'ye hazırlanan bir öğrenciye koçluk yapan sabırlı bir matematik/fen öğretmenisin. "
    "Öğrenci soruyu yanlış cevapladı. Türkçe ve kısa yaz. Şu başlıkları kullan:\n"
    "1) İpucu: cevabı söylemeden, öğrencinin tekrar denemesini sağlayacak tek bir yönlendirme.\n"
    "2) Çözüm: adım adım, kısa.\n"
    "3) Neden '{student}' şıkkı cazip geldi: bu şıkka götüren olası hata/kavram yanılgısı.\n"
    "4) Benzer soru: aynı kavramı ölçen yeni bir soru (şıklarıyla, cevabı en sonda)."
)

JUDGE_PROMPT = (
    "You are grading tutoring feedback for a Turkish university-entrance-exam question. "
    "The correct answer is {gold}; the student chose {student}. Score each 1-5: "
    "correctness (solution is mathematically right and reaches {gold}), "
    "misconception (plausibly explains why {student} is tempting), "
    "hint_quality (helps without giving away the answer), practice_question (valid and on-topic). "
    'Reply with JSON only: {{"correctness":n,"misconception":n,"hint_quality":n,"practice_question":n}}'
)


def _question_text(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(item["choices"]))
    return f"Soru:\n{item['question']}\n\nŞıklar:\n{opts}"


async def feedback(model: Model, item: dict, student: str) -> str:
    msgs = [{"role": "system", "content": TUTOR_PROMPT.format(student=student)},
            {"role": "user", "content": f"{_question_text(item)}\n\nDoğru cevap: {item['answer']}\n"
                                        f"Öğrencinin cevabı: {student}"}]
    r = await model.chat(msgs)
    if r.error:
        raise RuntimeError(r.error)
    print(f"[{model.name}] {r.latency_s:.1f}s, {r.out_tokens} out tokens, ${r.cost_usd:.5f}\n")
    return r.text or ""


async def judge(model: Model, item: dict, student: str, text: str) -> dict:
    msgs = [{"role": "system", "content": JUDGE_PROMPT.format(gold=item["answer"], student=student)},
            {"role": "user", "content": f"{_question_text(item)}\n\n--- FEEDBACK ---\n{text}"}]
    r = await model.chat(msgs)
    raw = (r.text or "").strip().strip("`").removeprefix("json").strip()
    try:
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except ValueError:
        return {"unparsed": r.text}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--student", required=True, choices=list(LETTERS))
    ap.add_argument("--model", default="nemotron-lightning")
    ap.add_argument("--judge")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    cfg = load_config()
    item = next(i for i in load(a.split) if i["id"] == a.id)
    text = await feedback(Model.from_config(cfg, a.model), item, a.student)
    print(text)
    if a.judge:
        print("\nJudge:", await judge(Model.from_config(cfg, a.judge), item, a.student, text))


if __name__ == "__main__":
    asyncio.run(main())
