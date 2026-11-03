"""Tutor app: a small-model tutor with a human in the loop.

    streamlit run app.py

Student tab   answer -> (wrong) diagnosis + misconception -> hint ladder -> retry -> targeted, verified practice
Teacher tab   approve / reject / correct each diagnosis -> export SFT + DPO training data
Class tab     topic mastery + most common misconceptions across all students
Cost tab      unit economics from measured per-call cost/latency (small model vs frontier)
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re

import pandas as pd
import streamlit as st

from yks import store
from yks.client import load_config
from yks.prompts import LETTERS
from yks.tutor import Tutor, load_pool

st.set_page_config(page_title="Tiny Tutors", page_icon="🎓", layout="wide")
CFG = load_config()
FRONTIER = "claude-opus-5.5"  # price reference for the "vs frontier" estimate
LANG = os.environ.get("TUTOR_LANG", "en")  # tutor explanations: "en" (default) or "tr"

MAX_CALLS = int(os.environ.get("MAX_CALLS_PER_SESSION", "0"))  # 0 = unlimited


def over_budget() -> bool:
    return bool(MAX_CALLS) and len(st.session_state.get("calls", [])) >= MAX_CALLS


def run(coro):
    return asyncio.run(coro)


def tex(s: str) -> str:
    """TurkishMMLU wraps math as $\\( ... \\)$; Streamlit markdown wants $...$."""
    s = s.replace("$\\(", "$").replace("\\)$", "$").replace("\\(", "$").replace("\\)", "$")
    return re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)


def available_models() -> list[str]:
    out = []
    for name, m in CFG["models"].items():
        prov = CFG["providers"][m["provider"]]
        if prov.get("type") == "bedrock_converse" or os.environ.get(prov.get("api_key_env", ""), ""):
            if "YOUR_ACCOUNT" not in m["model"]:
                out.append(name)
    return out


def frontier_cost(in_tok: int, out_tok: int) -> float:
    m = CFG["models"][FRONTIER]
    return (in_tok * m["price_in"] + out_tok * m["price_out"]) / 1e6


def track(kind: str, res: dict) -> None:
    st.session_state.calls.append({"kind": kind, "model": st.session_state.model, "cost": res["cost_usd"],
                                   "latency": res["latency_s"], "in": res["in_tokens"], "out": res["out_tokens"]})


def call_badge(res: dict) -> None:
    st.caption(f"⏱ {res['latency_s']:.1f}s · 💲{res['cost_usd']:.5f} · {res['out_tokens']} output tokens · "
               f"model `{st.session_state.model}` · same tokens at {FRONTIER} list price ≈ "
               f"${frontier_cost(res['in_tokens'], res['out_tokens']):.5f} (estimate)")


# ---------------- state ----------------
ss = st.session_state
ss.setdefault("calls", [])
ss.setdefault("q", None)
ss.setdefault("phase", "answer")
ss.setdefault("seen", set())
POOL = load_pool()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("🎓 Tiny Tutors")
    ss.student = st.text_input("Student (pseudonym)", value=ss.get("student", "student-1"))
    models = available_models()
    default = "gpt-oss-120b" if "gpt-oss-120b" in models else models[0]
    ss.model = st.selectbox("Tutor model", models, index=models.index(ss.get("model", default))
                            if ss.get("model", default) in models else 0)
    st.divider()
    st.subheader("This session")
    if ss.calls:
        df = pd.DataFrame(ss.calls)
        c1, c2 = st.columns(2)
        c1.metric("Model calls", len(df))
        c2.metric("Avg latency", f"{df.latency.mean():.1f}s")
        spent = df.cost.sum()
        fr = sum(frontier_cost(r["in"], r["out"]) for r in ss.calls)
        c1.metric("Spent", f"${spent:.4f}")
        c2.metric(f"At {FRONTIER} price", f"${fr:.4f}",
                  delta=f"{fr / spent:.0f}× more" if spent else None, delta_color="inverse")
    else:
        st.caption("No model calls yet.")
    st.divider()
    st.subheader("Weakness map")
    m = store.mastery(ss.student)
    if m:
        st.dataframe(pd.DataFrame(m)[["topic", "n", "ok", "mastery"]].rename(
            columns={"n": "tries", "ok": "first-try ✓"}), hide_index=True, width="stretch")
    else:
        st.caption("Answer a few questions to build it.")

tab_s, tab_t, tab_c, tab_b = st.tabs(["🧑‍🎓 Student", "👩‍🏫 Teacher", "📊 Class", "💼 Cost"])


# ---------------- student ----------------
def pick_next() -> None:
    """Recommend: weakest topic first (mastery < 0.6), otherwise an unseen random question."""
    weak = [r for r in store.mastery(ss.student) if r["mastery"] < 0.6 and r["topic"] != "?"]
    cands = []
    if weak:
        cands = [i for i in POOL if i.get("topic") == weak[0]["topic"] and i["id"] not in ss.seen]
        ss.reason = f"Recommended: your weakest topic **{weak[0]['topic']}** (mastery {weak[0]['mastery']:.2f})"
    if not cands:
        cands = [i for i in POOL if i["id"] not in ss.seen] or POOL
        ss.reason = "Random question from the practice pool"
    set_question(random.choice(cands))


def set_question(item: dict) -> None:
    ss.q, ss.phase, ss.diag, ss.hints, ss.attempt_id, ss.retry = item, "answer", None, 0, None, None
    ss.seen.add(item["id"])


with tab_s:
    top = st.columns([1, 1, 3])
    if top[0].button("▶ Next question", type="primary") or ss.q is None:
        pick_next()
    weak = [r for r in store.mastery(ss.student) if r["topic"] != "?"]
    if over_budget():
        st.warning(f"Session limit reached ({MAX_CALLS} model calls). Reload the page to start a new session.")
    if weak and not over_budget() and top[1].button("🎯 Practice weakest topic"):
        mis = store.misconceptions(ss.student, 1)
        with st.spinner("Generating and verifying a targeted question…"):
            res = run(Tutor(ss.model, CFG, lang=LANG).practice(weak[0]["topic"],
                                                    mis[0]["misconception"] if mis else "temel kavram hatası"))
        track("practice", res)
        if res["item"]:
            store.save_generated(res["item"])
            set_question(res["item"])
            ss.reason = f"Generated for **{weak[0]['topic']}** (verified by an independent solve ✓)"
        else:
            st.warning("Couldn't produce a verified question this time. Showing a pool question instead.")
            pick_next()

    q = ss.q
    st.caption(ss.get("reason", ""))
    st.markdown(f"**{q.get('subject', '')}** · {q.get('topic', '?')} · `{q['id']}` ({q.get('source', '?')})")
    st.markdown(tex(q["question"]))
    tr_key = f"tr-{q['id']}"
    if q.get("source") != "generated":  # pool questions are Turkish; generated practice is already English
        if tr_key in ss:
            st.info(f"🌐 **English translation** (machine-translated):\n\n{tex(ss[tr_key])}")
        elif not over_budget() and st.button("🌐 Translate to English", key=f"btn-{tr_key}"):
            with st.spinner("Translating…"):
                res = run(Tutor(ss.model, CFG, lang=LANG).translate(q))
            track("translate", res)
            ss[tr_key] = res["text"]
            st.rerun()
    choice = st.radio("Your answer", list(LETTERS), horizontal=True, index=None,
                      format_func=lambda L: f"{L}) {tex(q['choices'][LETTERS.index(L)])}",
                      key=f"ans-{q['id']}", disabled=ss.phase != "answer")

    if ss.phase == "answer" and st.button("Submit answer", disabled=choice is None):
        if choice == q["answer"]:
            store.log_attempt(ss.student, q, choice)
            ss.phase = "correct"
        elif over_budget():
            st.warning("Session limit reached; showing the answer key instead.")
            store.log_attempt(ss.student, q, choice)
            st.info(f"Correct answer: {q['answer']}")
            st.stop()
        else:
            with st.spinner("Diagnosing your mistake…"):
                ss.diag = run(Tutor(ss.model, CFG, lang=LANG).diagnose(q, choice))
            track("diagnose", ss.diag)
            ss.attempt_id = store.log_attempt(ss.student, q, choice, ss.diag)
            ss.chosen, ss.phase = choice, "wrong"
        st.rerun()

    if ss.phase == "correct":
        st.success(f"✅ Correct! ({q['answer']})")

    if ss.phase in ("wrong", "revealed"):
        d = ss.diag
        st.error(f"❌ Not {ss.chosen}. Let's see why.")
        call_badge(d)
        if not d["consistent"]:
            st.warning("⚠️ The tutor's own solution didn't reach the answer key. Flagged for teacher review.")
        st.markdown(f"**Likely mistake:** {d['misconception']}  \n**Why {ss.chosen} was tempting:** {d['why_tempting']}")
        st.markdown(f"**Topic:** {d['topic']}")
        for i, h in enumerate(d["hints"][:ss.hints]):
            st.info(f"💡 Hint {i + 1}: {h}")
        cols = st.columns(3)
        if ss.phase == "wrong" and ss.hints < len(d["hints"]) and cols[0].button(f"💡 Hint {ss.hints + 1}"):
            ss.hints += 1
            store.update_hints(ss.attempt_id, ss.hints, False)
            st.rerun()
        if ss.phase == "wrong" and ss.hints:
            retry = cols[1].selectbox("Try again", ["—"] + list(LETTERS), key=f"retry-{q['id']}-{ss.hints}")
            if retry != "—":
                ok = retry == q["answer"]
                store.update_hints(ss.attempt_id, ss.hints, ok)
                (st.success if ok else st.warning)(
                    f"{'✅ Correct! You solved it with a hint.' if ok else '❌ Not yet. Take another hint.'}")
                if ok:
                    ss.phase = "revealed"
        if cols[2].button("📖 Show solution"):
            ss.phase = "revealed"
            st.rerun()
        if ss.phase == "revealed":
            st.markdown(f"**Solution** (correct answer {q['answer']}):\n\n{tex(d['solution'])}")


# ---------------- teacher ----------------
with tab_t:
    st.subheader("Review queue")
    st.caption("Every diagnosis lands here. Approve it, reject it, or correct it. Corrections become DPO "
               "pairs (teacher preferred over model) and approvals become SFT rows for the next tutor model.")
    pending = store.reviews("pending")
    st.write(f"{len(pending)} pending")
    for r in pending[:10]:
        diag = json.loads(r["diagnosis_json"])
        with st.expander(f"#{r['id']} · {r['qid']} · student {r['student']} chose {r['chosen']} (key {r['gold']}) · "
                         f"{'✓ consistent' if r['consistent'] else '⚠️ inconsistent'} · {r['model']}"):
            st.json(diag)
            edited = st.text_area("Correct it (keep the JSON shape)", value=json.dumps(diag, ensure_ascii=False,
                                                                                     indent=2), key=f"edit-{r['id']}")
            c = st.columns(3)
            if c[0].button("✅ Approve", key=f"ok-{r['id']}"):
                store.set_review(r["id"], "approved")
                st.rerun()
            if c[1].button("✏️ Save correction", key=f"fix-{r['id']}"):
                store.set_review(r["id"], "corrected", edited)
                st.rerun()
            if c[2].button("🗑 Reject", key=f"no-{r['id']}"):
                store.set_review(r["id"], "rejected")
                st.rerun()
    st.divider()
    done = store.reviews(status=None)
    stats = pd.Series([r["status"] for r in done]).value_counts().to_dict() if done else {}
    st.write("Review outcomes:", stats or "none yet")
    if st.button("⬇️ Export training data for Fireworks"):
        out = store.export_training()
        st.success(f"Wrote {out['sft']} SFT rows and {out['dpo']} DPO pairs to `{out['dir']}`")
        st.code("firectl dataset create yks-tutor-dpo data/feedback/tutor_dpo.jsonl\n"
                "firectl dpo-job create --base-model <TUTOR_MODEL> --dataset yks-tutor-dpo "
                "--output-model yks-tutor-dpo-v2   # check `firectl dpo-job create --help`", language="bash")

# ---------------- class insights ----------------
with tab_c:
    st.subheader("Class view (all students)")
    m_all = store.mastery()
    if m_all:
        df = pd.DataFrame(m_all)
        df["first-try accuracy"] = df.ok / df.n
        st.bar_chart(df.set_index("topic")["mastery"])
        st.dataframe(df[["topic", "n", "ok", "first-try accuracy", "hints", "mastery"]], hide_index=True,
                     width="stretch")
        st.subheader("Most common misconceptions")
        st.dataframe(pd.DataFrame(store.misconceptions(limit=15)), hide_index=True, width="stretch")
    else:
        st.caption("No attempts yet.")

# ---------------- cost ----------------
with tab_b:
    st.subheader("Unit economics (estimates from measured calls)")
    usage = store.usage_by_model()
    if usage:
        u = pd.DataFrame(usage)
        u["same tokens at frontier price"] = [frontier_cost(r["avg_in"], r["avg_out"]) for r in usage]
        st.dataframe(u, hide_index=True, width="stretch")
    by_model = {r["model"]: r for r in usage}
    c = st.columns(3)
    students = c[0].number_input("Students", 1_000, 5_000_000, 100_000, step=10_000)
    qpd = c[1].number_input("Questions / student / day", 1, 200, 30)
    days = c[2].number_input("Study days / year", 30, 365, 250)
    wrong = c[0].slider("Wrong-answer rate (each triggers a diagnosis)", 0.05, 0.9, 0.4)
    small = c[1].selectbox("Small tutor model", list(by_model) or ["nemotron-lightning"])
    sm = by_model.get(small)
    small_cost = c[2].number_input("$ per diagnosis (small)", 0.0, 1.0,
                                   float(sm["avg_cost"]) if sm else 0.0005, format="%.5f")
    fr_cost = c[2].number_input(f"$ per diagnosis ({FRONTIER})", 0.0, 1.0,
                                float(frontier_cost(sm["avg_in"], sm["avg_out"])) if sm else 0.015, format="%.5f")
    diagnoses = students * qpd * days * wrong
    k = st.columns(4)
    k[0].metric("Diagnoses / year", f"{diagnoses:,.0f}")
    k[1].metric("Small model / year", f"${diagnoses * small_cost:,.0f}")
    k[2].metric(f"{FRONTIER} / year", f"${diagnoses * fr_cost:,.0f}")
    k[3].metric("Per student / year (small)", f"${diagnoses * small_cost / students:,.2f}")
    st.caption("The frontier column prices the small model's measured token counts at frontier list price. "
               "Frontier models often use more tokens, so treat it as a lower bound. For the real comparison "
               "use `python -m yks.tutor_eval`, which runs both models on the same cases.")
