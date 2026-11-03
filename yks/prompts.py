"""Prompt construction and robust answer extraction (same for every system, so the comparison is fair).

Two modes:
  standard  always answer A-E (pure accuracy)
  exam      real YKS scoring: +1 correct, -1/4 wrong, 0 blank. The model may answer 'BOŞ' (blank).
            This lets us measure calibrated abstention: does the model know when it doesn't know?
"""
from __future__ import annotations

import re

LETTERS = "ABCDE"
BLANK = "BOŞ"

SYSTEM_PROMPT = (
    "Sen Türkiye'deki üniversite giriş sınavı (YKS: TYT/AYT) sorularını çözen uzman bir öğretmensin. "
    "Soruyu adım adım çöz, sonucunu şıklarla karşılaştır. "
    "Yanıtının EN SON satırı tam olarak şu biçimde olmalı: 'Cevap: X' (X ∈ {A,B,C,D,E})."
)

EXAM_PROMPT = (
    "Sen Türkiye'deki üniversite giriş sınavı (YKS: TYT/AYT) sorularını çözen uzman bir öğretmensin. "
    "Puanlama gerçek sınavdaki gibidir: doğru cevap +1, yanlış cevap -0,25 (4 yanlış 1 doğruyu götürür), "
    "boş bırakılan soru 0 puan. Soruyu adım adım çöz. Cevabından emin değilsen boş bırakmak daha iyidir. "
    "Yanıtının EN SON satırı tam olarak şu biçimde olmalı: 'Cevap: X' (X ∈ {A,B,C,D,E}) "
    "ya da boş bırakıyorsan 'Cevap: BOŞ'."
)

PROMPTS = {"standard": SYSTEM_PROMPT, "exam": EXAM_PROMPT}

# Free-response (olympiad) prompt: English, matching the competitions' language and the common
# MathArena-style convention of a single final \boxed{} answer.
FREE_PROMPT = (
    "You are an expert competition mathematician. Solve the problem step by step, checking your work. "
    "Give the final answer in simplest exact form (no decimals unless the problem asks for them) as the "
    "LAST thing in your response, inside \\boxed{}."
)


def format_question(item: dict) -> str:
    if not item.get("choices"):
        return f"Problem:\n{item['question']}"
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(item["choices"]))
    return f"Soru:\n{item['question']}\n\nŞıklar:\n{opts}"


def build_messages(item: dict, mode: str = "standard") -> list[dict]:
    system = PROMPTS[mode] if item.get("choices") else FREE_PROMPT
    return [{"role": "system", "content": system}, {"role": "user", "content": format_question(item)}]


_ANS = r"([A-E]|BOŞ|BOS)\b"
_PATTERNS = [
    re.compile(r"(?:Cevap|Yanıt|Answer|Doğru\s+(?:cevap|şık))\s*[:：=]?\s*\**\(?\s*" + _ANS, re.IGNORECASE),
    re.compile(r"\\boxed\{\s*\(?([A-E])\)?\s*\}"),
    re.compile(r"\b([A-E])\)\s*$", re.MULTILINE),
]


def extract_answer(text: str | None) -> str | None:
    """Return the final A-E letter, 'BOŞ' for an explicit blank, or None if unparseable
    (counted as wrong, reported separately)."""
    if not text:
        return None
    # Ignore hidden reasoning if a model inlines it.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for pat in _PATTERNS:
        hits = pat.findall(text)
        if hits:
            h = hits[-1].upper()
            return BLANK if h in ("BOŞ", "BOS") else h
    return None
