"""Answer extraction + grading for both task types, so every system is scored identically.

mc    5-option multiple choice (YKS): letter A-E, or BOŞ (blank) in exam mode
free  final-answer problems (olympiad): last \\boxed{...}; compared to the key with math-verify
      (symbolic equivalence: 2\\sqrt{3} == \\sqrt{12}, -1/21 == -\\frac{1}{21}). Integer-answer
      problems (AIME) use exact integer match.
"""
from __future__ import annotations

import re
from collections import Counter

from .prompts import extract_answer as extract_mc


def is_mc(item: dict) -> bool:
    return bool(item.get("choices"))


def last_boxed(text: str | None) -> str | None:
    """Content of the last \\boxed{...} (brace-matched), else a trailing 'Final answer: ...' line."""
    if not text:
        return None
    text = text.rsplit("</think>", 1)[-1]
    i = text.rfind("\\boxed")
    if i >= 0:
        j = text.find("{", i)
        if j >= 0:
            depth, k = 0, j
            while k < len(text):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    return text[j + 1:k].strip() or None
                k += 1
    m = re.findall(r"(?:final answer|answer)\s*[:：]\s*\$?([^\n$]+)", text, re.IGNORECASE)
    return m[-1].strip() if m else None


def extract(item: dict, text: str | None) -> str | None:
    return extract_mc(text) if is_mc(item) else last_boxed(text)


def _as_int(s: str) -> int | None:
    s = re.sub(r"[\s,$]|\\!|\\,|^\\text\{|\}$", "", s)
    s = re.sub(r"^0+(?=\d)", "", s)  # AIME answers are written 000-999
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def equivalent(item: dict, a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if is_mc(item):
        return a == b
    if item.get("answer_type") == "integer":
        ia, ib = _as_int(a), _as_int(b)
        if ia is not None and ib is not None:
            return ia == ib
    if a.strip() == b.strip():
        return True
    from math_verify import parse, verify
    try:
        return bool(verify(parse(f"${a}$"), parse(f"${b}$")))
    except Exception:  # noqa: BLE001 - unparseable LaTeX is simply "not equivalent"
        return False


def is_correct(item: dict, pred: str | None) -> bool:
    return equivalent(item, item["answer"], pred)


def vote(item: dict, preds: list[str | None]) -> tuple[str | None, float]:
    """Majority vote. For free-form answers, equivalent answers (e.g. 1/2 and 0.5... or \\frac12)
    are merged into one cluster before counting."""
    valid = [p for p in preds if p]
    if not valid:
        return None, 0.0
    if is_mc(item):
        letter, n = Counter(valid).most_common(1)[0]
        return letter, n / len(preds)
    clusters: list[list[str]] = []
    for p in valid:
        for c in clusters:
            if equivalent(item, c[0], p):
                c.append(p)
                break
        else:
            clusters.append([p])
    best = max(clusters, key=len)
    return best[0], len(best) / len(preds)
