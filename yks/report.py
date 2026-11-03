"""Aggregate results into a comparison table + Pareto plots, with paired significance vs the baseline.

    python -m yks.report --split test --baseline claude-opus-5.5
    python -m yks.report --dataset olympiad --split test --baseline claude-opus-5.5
Writes results/<dataset>/<split>/report.md and pareto.png
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAM_LEN = 40


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r  # last row wins (retries)
    return rows


def bootstrap_ci(xs: list[int], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = system right & baseline wrong, c = the reverse."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def dedicated_cost_per_q(rdir: Path, name: str) -> tuple[float, str | None]:
    """GPU-hour cost per question for systems that use dedicated (billing: gpu_hour) models.

    cost/q = sum(gpus * $/GPU-hr) * measured wall hours / questions run. This is an ESTIMATE that
    depends on utilisation: higher concurrency -> more questions per GPU-hour -> lower cost.
    """
    cfg_path, runs_path = rdir / f"{name}.config.json", rdir / f"{name}.runs.jsonl"
    if not cfg_path.exists() or not runs_path.exists():
        return 0.0, None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    used = [cfg.get(k) for k in ("model", "small", "large") if cfg.get(k)]
    rate = sum(cfg["models"][m].get("gpu_hour_usd", 0) * cfg["models"][m].get("gpus", 1)
               for m in used if cfg["models"].get(m, {}).get("billing") == "gpu_hour")
    if not rate:
        return 0.0, None
    runs = [json.loads(l) for l in runs_path.open(encoding="utf-8") if l.strip()]
    n, wall = sum(r["n"] for r in runs), sum(r["wall_s"] for r in runs)
    qph = n / (wall / 3600)
    return rate * (wall / 3600) / n, (f"`{name}`: dedicated deployment at ${rate:.2f}/hr, measured "
                                      f"{qph:,.0f} questions/hr at concurrency "
                                      f"{runs[-1]['concurrency']} -> GPU-hour cost included")


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="yks")
    ap.add_argument("--systems", nargs="+", help="restrict to these systems")
    ap.add_argument("--min-complete", type=float, default=0.98)
    ap.add_argument("--baseline", default="claude-opus-5.5")
    ap.add_argument("--exclude-figures", action="store_true",
                    help="Also drop questions that reference an unseen figure (report both views).")
    a = ap.parse_args()

    rdir = ROOT / "results" / a.dataset / a.split
    yks = a.dataset == "yks"
    systems = {p.stem: load_rows(p) for p in sorted(rdir.glob("*.jsonl"))
               if not p.name.endswith(".runs.jsonl")}
    if a.systems:
        systems = {k: v for k, v in systems.items() if k in a.systems}
    # Skip systems that haven't finished (otherwise the identical-question intersection collapses).
    full = max(len(v) for v in systems.values())
    skipped = [k for k, v in systems.items() if len(v) < a.min_complete * full]
    for k in skipped:
        print(f"skipping {k}: {len(systems[k])}/{full} rows (incomplete)")
        systems.pop(k)
    if a.baseline not in systems:
        raise SystemExit(f"baseline '{a.baseline}' has no results in {rdir}")

    # Only compare on questions every system answered -> identical test cases.
    common = set.intersection(*(set(r) for r in systems.values()))
    if a.exclude_figures:
        common = {i for i in common if not systems[a.baseline][i]["has_figure_ref"]}
    ids = sorted(common)
    base = systems[a.baseline]

    table, notes = [], []
    for name, rows in systems.items():
        rs = [rows[i] for i in ids]
        corr = [int(r["correct"]) for r in rs]
        lo, hi = bootstrap_ci(corr)
        lat = [r["latency_s"] for r in rs]
        gpu_cost_q, note = dedicated_cost_per_q(rdir, name)
        if note:
            notes.append(note)
        cost = sum(r["cost_usd"] for r in rs) + gpu_cost_q * len(rs)
        b = sum(1 for i in ids if rows[i]["correct"] and not base[i]["correct"])
        c = sum(1 for i in ids if not rows[i]["correct"] and base[i]["correct"])
        by_subj, by_year = {}, {}
        for r in rs:
            by_subj.setdefault(r["subject"], []).append(int(r["correct"]))
            by_year.setdefault(r.get("year") or "?", []).append(int(r["correct"]))
        blanks = sum(1 for r in rs if r.get("abstained"))
        wrong = len(rs) - sum(corr) - blanks  # parse failures count as wrong (conservative)
        table.append({
            "system": name, "n": len(rs), "acc": sum(corr) / len(rs), "ci": (lo, hi),
            # YKS net = correct - wrong/4, scaled to a 40-question test (TYT math length)
            "net40": EXAM_LEN * (sum(corr) - wrong / 4) / len(rs), "blank": blanks / len(rs),
            "p50": statistics.median(lat), "p95": pct(lat, 0.95),
            "cost_per_1k": 1000 * cost / len(rs),
            "out_tok": statistics.mean(r["out_tokens"] for r in rs),
            "parse_fail": sum(r["parse_fail"] for r in rs),
            "api_err": sum(1 for r in rs if r["errors"]),
            "escalated": sum(r.get("escalated", False) for r in rs) / len(rs),
            "wins": b, "losses": c, "p": mcnemar_exact(b, c),
            "subj": {k: sum(v) / len(v) for k, v in by_subj.items()},
            "year": {k: (sum(v) / len(v), len(v)) for k, v in by_year.items()},
        })
    table.sort(key=lambda t: t["cost_per_1k"])
    bt = next(t for t in table if t["system"] == a.baseline)

    lines = [f"# Results ({a.dataset} / {a.split}, n={len(ids)} identical {'questions' if yks else 'problem runs'}"
             f"{', figure-referencing excluded' if a.exclude_figures else ''})", "",
             f"Baseline: `{a.baseline}`. Cost is estimated from measured token usage x list price "
             "in configs/systems.yaml. Latency is wall-clock per question at the run's concurrency. "
             + ("net/40 = YKS net (correct - wrong/4) scaled to a 40-question test; blanks only occur in exam-mode systems."
                if yks else "Olympiad: final answers graded with math-verify. With --repeats, each problem counts R times "
                "(CI is slightly optimistic)."), "",
             ("| system | acc | 95% CI | net/40 | blank | " if yks else "| system | acc | 95% CI | ")
             + "Δacc vs base | wins/losses vs base | McNemar p | p50 s | p95 s | "
             + ("$/40-q exam" if yks else "$/problem") + " | cost vs base | avg out tok | escalated | parse fail | api err |",
             "|---|---|---|" + ("---|---|" if yks else "") + "---|" * 11]
    for t in table:
        ratio = t["cost_per_1k"] / bt["cost_per_1k"] if bt["cost_per_1k"] else float("nan")
        lines.append(
            f"| {t['system']} | {t['acc']:.1%} | {t['ci'][0]:.1%}–{t['ci'][1]:.1%} | "
            + (f"{t['net40']:.1f} | {t['blank']:.0%} | " if yks else "") +
            f""
            f"{(t['acc'] - bt['acc']) * 100:+.1f} pp | {t['wins']}/{t['losses']} | {t['p']:.3f} | "
            f"{t['p50']:.1f} | {t['p95']:.1f} | ${t['cost_per_1k'] * (EXAM_LEN if yks else 1) / 1000:.4f} | {ratio:.2f}x | "
            f"{t['out_tok']:.0f} | {t['escalated']:.0%} | {t['parse_fail']} | {t['api_err']} |")
    if notes:
        lines += ["", "Cost notes:"] + [f"- {n}" for n in notes]
    subjects = sorted({s for t in table for s in t["subj"]})
    lines += ["", "## Accuracy by " + ("subject" if yks else "competition"), "", "| system | " + " | ".join(subjects) + " |",
              "|---|" + "---|" * len(subjects)]
    for t in table:
        lines.append(f"| {t['system']} | " + " | ".join(f"{t['subj'].get(s, 0):.1%}" for s in subjects) + " |")
    years = sorted({y for t in table for y in t["year"]}, key=str)
    if len(years) > 1:
        # A model that does much better on older contests than newer ones may have seen them in training.
        lines += ["", "## Accuracy by contest year (contamination check)", "",
                  "| system | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years)]
        for t in table:
            lines.append(f"| {t['system']} | " + " | ".join(
                f"{t['year'][y][0]:.1%} (n={t['year'][y][1]})" if y in t["year"] else "-" for y in years) + " |")
    # Is self-agreement a usable confidence signal? (justifies cascade routing)
    lines += ["", "## Accuracy by vote agreement (multi-sample systems)", "",
              "| system | agreement | n | acc |", "|---|---|---|---|"]
    for name, rows in systems.items():
        rs = [rows[i] for i in ids]
        if not rs or len(rs[0].get("votes", [])) < 2:
            continue
        buckets = {}
        for r in rs:
            buckets.setdefault(round(r["agreement"], 2), []).append(int(r["correct"]))
        for agr in sorted(buckets, reverse=True):
            v = buckets[agr]
            lines.append(f"| {name} | {agr:.0%} | {len(v)} | {sum(v) / len(v):.1%} |")
    suffix = "_nofig" if a.exclude_figures else ""
    (rdir / f"report{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in ((axes[0], "cost_per_1k", "Est. cost per 1,000 questions (USD, log)"),
                           (axes[1], "p50", "Median latency per question (s, log)")):
        for t in table:
            color = "crimson" if t["system"] == a.baseline else "tab:blue"
            ax.errorbar(t[key], t["acc"], yerr=[[t["acc"] - t["ci"][0]], [t["ci"][1] - t["acc"]]],
                        fmt="o", color=color, capsize=3)
            ax.annotate(t["system"], (t[key], t["acc"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{'YKS STEM' if yks else 'Olympiad'} accuracy vs efficiency (n={len(ids)}, red = frontier baseline)")
    fig.tight_layout()
    fig.savefig(rdir / f"pareto{suffix}.png", dpi=150)
    print(f"saved {rdir / f'pareto{suffix}.png'}")


if __name__ == "__main__":
    main()
