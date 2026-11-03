"""SQLite store for the human-in-the-loop layer.

attempts  every student answer (topic, misconception, hints used, model cost/latency)
reviews   every model diagnosis, queued for a teacher: approve / reject / correct
export    approved -> SFT rows; corrected -> DPO pairs (teacher text preferred over model text)

This is how the system "learns from humans": students' mistakes drive what they practise next, and
teachers' corrections become Fireworks training data for the next tutor model.
The DB holds student data: it's gitignored. Use pseudonyms.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "app.db"
EXPORT = ROOT / "data" / "feedback"

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY, ts REAL, student TEXT, qid TEXT, source TEXT, subject TEXT, topic TEXT,
  chosen TEXT, gold TEXT, correct INTEGER, hints_used INTEGER DEFAULT 0, solved_after_hints INTEGER DEFAULT 0,
  misconception TEXT, model TEXT, cost_usd REAL DEFAULT 0, latency_s REAL DEFAULT 0,
  in_tokens INTEGER DEFAULT 0, out_tokens INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY, ts REAL, attempt_id INTEGER, qid TEXT, model TEXT, consistent INTEGER,
  prompt_json TEXT, diagnosis_json TEXT, status TEXT DEFAULT 'pending', teacher_text TEXT);
CREATE TABLE IF NOT EXISTS generated (id TEXT PRIMARY KEY, ts REAL, item_json TEXT);
"""


def conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def log_attempt(student: str, item: dict, chosen: str, diag: dict | None = None) -> int:
    d = diag or {}
    with conn() as c:
        cur = c.execute(
            "INSERT INTO attempts (ts, student, qid, source, subject, topic, chosen, gold, correct, misconception,"
            " model, cost_usd, latency_s, in_tokens, out_tokens) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), student, item["id"], item.get("source", "?"), item.get("subject"),
             d.get("topic") or item.get("topic") or "?", chosen, item["answer"], int(chosen == item["answer"]),
             d.get("misconception"), d.get("model"), d.get("cost_usd", 0), d.get("latency_s", 0),
             d.get("in_tokens", 0), d.get("out_tokens", 0)))
        attempt_id = cur.lastrowid
        if diag:
            c.execute("INSERT INTO reviews (ts, attempt_id, qid, model, consistent, prompt_json, diagnosis_json)"
                      " VALUES (?,?,?,?,?,?,?)",
                      (time.time(), attempt_id, item["id"], d.get("model"), int(bool(d.get("consistent"))),
                       json.dumps(d.get("prompt_messages"), ensure_ascii=False),
                       json.dumps({k: d.get(k) for k in ("topic", "misconception", "why_tempting", "hints",
                                                         "solution")}, ensure_ascii=False)))
    return attempt_id


def update_hints(attempt_id: int, hints_used: int, solved: bool) -> None:
    with conn() as c:
        c.execute("UPDATE attempts SET hints_used=?, solved_after_hints=? WHERE id=?",
                  (hints_used, int(solved), attempt_id))


def save_generated(item: dict) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO generated VALUES (?,?,?)",
                  (item["id"], time.time(), json.dumps(item, ensure_ascii=False)))


def mastery(student: str | None = None) -> list[dict]:
    """Per-topic mastery = (first-try correct + 1) / (attempts + 2) (Laplace-smoothed).
    Lowest score = weakest topic = what we recommend practising next."""
    q = "SELECT topic, COUNT(*) n, SUM(correct) ok, SUM(hints_used) hints FROM attempts"
    args: tuple = ()
    if student:
        q += " WHERE student=?"
        args = (student,)
    q += " GROUP BY topic"
    with conn() as c:
        rows = [dict(r) for r in c.execute(q, args)]
    for r in rows:
        r["mastery"] = (r["ok"] + 1) / (r["n"] + 2)
    return sorted(rows, key=lambda r: r["mastery"])


def misconceptions(student: str | None = None, limit: int = 10) -> list[dict]:
    q = ("SELECT topic, misconception, COUNT(*) n FROM attempts WHERE correct=0 AND misconception IS NOT NULL")
    args: tuple = ()
    if student:
        q += " AND student=?"
        args = (student,)
    q += " GROUP BY topic, misconception ORDER BY n DESC LIMIT ?"
    with conn() as c:
        return [dict(r) for r in c.execute(q, args + (limit,))]


def usage_by_model() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT model, COUNT(*) calls, AVG(cost_usd) avg_cost, AVG(latency_s) avg_latency,"
            " AVG(in_tokens) avg_in, AVG(out_tokens) avg_out FROM attempts WHERE model IS NOT NULL GROUP BY model")]


def reviews(status: str | None = "pending") -> list[dict]:
    q = "SELECT r.*, a.chosen, a.gold, a.student FROM reviews r JOIN attempts a ON a.id=r.attempt_id"
    args: tuple = ()
    if status:
        q += " WHERE r.status=?"
        args = (status,)
    with conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY r.ts DESC", args)]


def set_review(review_id: int, status: str, teacher_text: str | None = None) -> None:
    with conn() as c:
        c.execute("UPDATE reviews SET status=?, teacher_text=? WHERE id=?", (status, teacher_text, review_id))


def export_training() -> dict:
    """approved -> SFT (model output is good); corrected -> DPO (teacher text preferred) + SFT on teacher text."""
    EXPORT.mkdir(parents=True, exist_ok=True)
    n_sft = n_dpo = 0
    with open(EXPORT / "tutor_sft.jsonl", "w", encoding="utf-8") as sft, \
         open(EXPORT / "tutor_dpo.jsonl", "w", encoding="utf-8") as dpo:
        for r in reviews(status=None):
            prompt = json.loads(r["prompt_json"] or "null")
            if not prompt:
                continue
            model_out = r["diagnosis_json"]
            if r["status"] == "approved":
                sft.write(json.dumps({"messages": prompt + [{"role": "assistant", "content": model_out}]},
                                     ensure_ascii=False) + "\n")
                n_sft += 1
            elif r["status"] == "corrected" and r["teacher_text"]:
                sft.write(json.dumps({"messages": prompt + [{"role": "assistant", "content": r["teacher_text"]}]},
                                     ensure_ascii=False) + "\n")
                dpo.write(json.dumps({  # Fireworks DPO format
                    "input": {"messages": prompt},
                    "preferred_output": [{"role": "assistant", "content": r["teacher_text"]}],
                    "non_preferred_output": [{"role": "assistant", "content": model_out}]},
                    ensure_ascii=False) + "\n")
                n_sft += 1
                n_dpo += 1
    return {"sft": n_sft, "dpo": n_dpo, "dir": str(EXPORT)}
