"""The fine-tuning value chart: Fireworks-tunable small models vs frontier, before and after fine-tuning.

Left panel   score vs model size (B params, log). Base tunable models = circles (colour = family);
             fine-tuned models = gold diamonds with an arrow from their base and the "+X pp" gain;
             frontier baselines = dashed horizontal lines (their size isn't public).
Right panel  score vs measured cost per 1,000 questions (log). Frontier = red stars. The green zone is
             "at least frontier quality, for less money"; the fine-tuned model is annotated with its
             cost multiple vs the frontier baseline. This is the business-value view.

Points come from results/<dataset>/<split>/<system>.jsonl (same files as yks.report), restricted to
questions EVERY plotted system answered. A system is plotted if it has results and its model has
size_b in configs/systems.yaml (tunable ladder / fine-tuned) or is a frontier baseline (--frontier).

    python -m yks.plot_finetune --split test --frontier claude-opus-5.5 gpt-6.1-sol
    python -m yks.plot_finetune --split test --metric net40          # YKS net score instead of accuracy
    python -m yks.plot_finetune --dataset olympiad --split test --frontier claude-opus-5.5
    python -m yks.plot_finetune --demo                               # layout preview, MOCK data
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .client import load_config  # noqa: E402
from .report import EXAM_LEN, bootstrap_ci, dedicated_cost_per_q, load_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD, FRONTIER_COLORS = "#FFB000", ["#C0392B", "#7D3C98", "#1F618D", "#AF601A"]


# Candidate label offsets (points) tried in order: below, above, right, left, then further out.
_OFFSETS = [(0, -22), (0, 20), (24, 0), (-24, 0), (0, -38), (0, 36), (30, -24), (-30, -24), (30, 22),
            (-30, 22), (0, -54), (0, 52), (46, 0), (-46, 0)]


def place_labels(ax, labels: list[dict], markers: list[tuple[float, float, float]], obstacles=()) -> None:
    """Greedy de-overlap. labels: {x, y, text, kw, offsets?}; markers: (x, y, scatter size s in pt^2);
    obstacles: artists (e.g. captions) labels must also avoid.
    Each label takes the first offset whose box hits no marker and no already-placed label."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for (mx, my, size) in markers:
        X, Y = ax.transData.transform((mx, my))
        rad = (size ** 0.5) / 2 * fig.dpi / 72 + 3  # scatter s is area in pt^2 -> radius in px
        boxes.append((X - rad, Y - rad, X + rad, Y + rad))
    for art in obstacles:
        e = art.get_window_extent(r)
        boxes.append((e.x0, e.y0, e.x1, e.y1))
    def overlap(b) -> float:
        return sum(max(0, min(b[2], o[2]) - max(b[0], o[0])) * max(0, min(b[3], o[3]) - max(b[1], o[1]))
                   for o in boxes)

    ab = ax.get_window_extent(r)
    for lab in labels:
        best = None  # (overlap score, offset)
        for dx, dy in lab.get("offsets", _OFFSETS):
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
            t = ax.annotate(lab["text"], (lab["x"], lab["y"]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, zorder=6, **lab.get("kw", {}))
            e = t.get_window_extent(r)
            b = (e.x0 - 5, e.y0 - 4, e.x1 + 5, e.y1 + 4)  # fixed px margin between labels
            t.remove()
            outside = (max(0, ab.x0 - b[0]) + max(0, b[2] - ab.x1)) * (b[3] - b[1]) + \
                      (max(0, ab.y0 - b[1]) + max(0, b[3] - ab.y1)) * (b[2] - b[0])
            score = overlap(b) + 4 * outside
            if best is None or score < best[0]:
                best = (score, dx, dy, ha, va, b)
            if score == 0:
                break
        _, dx, dy, ha, va, b = best  # first free spot, else the least-overlapping one
        kw = dict(lab.get("kw", {}))
        if abs(dx) + abs(dy) > 30 and "arrowprops" not in kw:  # far from its point: draw a leader line
            kw["arrowprops"] = dict(arrowstyle="-", color="#888", lw=0.8, shrinkA=2, shrinkB=9)
        ax.annotate(lab["text"], (lab["x"], lab["y"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, zorder=6, **kw)
        boxes.append(b)


def _metric(rows: list[dict], metric: str) -> tuple[float, tuple[float, float]]:
    corr = [int(r["correct"]) for r in rows]
    if metric == "acc":
        lo, hi = bootstrap_ci(corr)
        return 100 * sum(corr) / len(corr), (100 * lo, 100 * hi)
    # YKS net per 40 questions: +1 correct, -1/4 wrong, 0 blank (parse failures count as wrong)
    per_q = [1.0 if r["correct"] else (0.0 if r.get("abstained") else -0.25) for r in rows]
    import random
    rng = random.Random(0)
    boots = sorted(sum(rng.choices(per_q, k=len(per_q))) / len(per_q) for _ in range(2000))
    return EXAM_LEN * sum(per_q) / len(per_q), (EXAM_LEN * boots[50], EXAM_LEN * boots[1949])


def collect(dataset: str, split: str, metric: str, frontier: list[str],
            only: list[str] | None = None) -> tuple[list[dict], int]:
    cfg = load_config()
    rdir = ROOT / "results" / dataset / split
    results = {p.stem: load_rows(p) for p in rdir.glob("*.jsonl") if not p.name.endswith(".runs.jsonl")}
    model_to_system = {}
    chosen = {}
    for name, rows in results.items():
        if only and name not in only and name not in frontier:
            continue
        spec = cfg["systems"].get(name)
        if not spec or spec.get("strategy") != "single":
            continue
        m = cfg["models"][spec["model"]]
        if name in frontier:
            chosen[name] = {"kind": "frontier", "label": name}
        elif m.get("size_b"):
            kind = "ft" if m.get("finetuned_from") else "base"
            chosen[name] = {"kind": kind, "label": m.get("label", name), "size_b": float(m["size_b"]),
                            "family": m.get("family", "?"), "base_model": m.get("finetuned_from")}
            if kind == "base":
                model_to_system[spec["model"]] = name
    if not chosen:
        raise SystemExit(f"No plottable systems with results in {rdir}. Run yks.run_eval first "
                         f"(tunable ladder + --frontier systems), or try --demo.")
    ids = sorted(set.intersection(*(set(results[n]) for n in chosen)))
    points = []
    for name, meta in chosen.items():
        rs = [results[name][i] for i in ids]
        val, ci = _metric(rs, metric)
        gpu_q, _ = dedicated_cost_per_q(rdir, name)
        cost = sum(r["cost_usd"] for r in rs) + gpu_q * len(rs)
        base_sys = model_to_system.get(meta.get("base_model")) or (
            meta.get("base_model") if meta.get("base_model") in chosen else None)
        points.append({**meta, "system": name, "value": val, "ci": ci, "cost_1k": 1000 * cost / len(rs),
                       "base_system": base_sys})
    return points, len(ids)


def demo_points() -> tuple[list[dict], int]:
    """MOCK numbers purely to preview the layout. Never present these."""
    fam = lambda n, s, f, v, c: {"kind": "base", "system": n, "label": n, "size_b": s, "family": f,  # noqa: E731
                                 "value": v, "ci": (v - 4, v + 4), "cost_1k": c, "base_system": None}
    pts = [fam("qwen3-0.6b", 0.75, "Qwen 3", 31, 0.06), fam("ministral-3b", 3.0, "Mistral", 44, 0.12),
           fam("llama3.2-3b", 3.61, "Llama", 39, 0.12), fam("qwen3-4b", 4.41, "Qwen 3", 58, 0.25),
           fam("qwen3-8b", 8.19, "Qwen 3", 66, 0.40), fam("llama3.1-8b", 8.84, "Llama", 49, 0.30),
           fam("qwen3.5-9b", 9.41, "Qwen 3.5", 71, 0.45), fam("qwen3-14b", 14.77, "Qwen 3", 74, 0.70),
           fam("qwen3-32b", 32.76, "Qwen 3", 81, 2.60),
           {"kind": "ft", "system": "yks-sft", "label": "Qwen3-8B + YKS SFT (ours)", "size_b": 8.19,
            "family": "Qwen 3", "value": 86, "ci": (82, 90), "cost_1k": 0.55, "base_system": "qwen3-8b"},
           {"kind": "frontier", "system": "claude-opus-5.5", "label": "claude-opus-5.5", "value": 88,
            "ci": (84, 92), "cost_1k": 14.0},
           {"kind": "frontier", "system": "gpt-6.1-sol", "label": "gpt-6.1-sol", "value": 85,
            "ci": (81, 89), "cost_1k": 5.2}]
    return pts, 240


def plot(points: list[dict], n: int, metric: str, dataset: str, split: str, out: Path,
         watermark: str | None = None) -> str:
    ylab = "Accuracy (%)" if metric == "acc" else "YKS net score per 40 questions"
    base = [p for p in points if p["kind"] == "base"]
    ft = [p for p in points if p["kind"] == "ft"]
    fr = sorted([p for p in points if p["kind"] == "frontier"], key=lambda p: -p["value"])
    by = {p["system"]: p for p in points}
    families = sorted({p["family"] for p in base + ft})
    fcol = {f: plt.cm.tab10(i % 10) for i, f in enumerate(families)}

    vals = [p["value"] for p in points]
    lo, hi = min(vals), max(vals)
    pad = max(3.0, 0.08 * (hi - lo))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2))
    for ax in (ax1, ax2):
        ax.grid(True, ls="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(lo - 2.5 * pad, hi + 2 * pad)
        ax.set_ylabel(ylab, fontsize=13, fontweight="bold")

    labels = {ax1: [], ax2: []}
    markers = {ax1: [], ax2: []}
    obstacles = {ax1: [], ax2: []}

    def circle(ax, p, x, label=True):
        markers[ax].append((x, p["value"], 230))
        ax.errorbar(x, p["value"], yerr=[[p["value"] - p["ci"][0]], [p["ci"][1] - p["value"]]],
                    fmt="none", ecolor=fcol[p["family"]], alpha=0.35, capsize=3)
        ax.scatter(x, p["value"], s=230, color=fcol[p["family"]], edgecolor="#333", lw=1.2, alpha=0.85, zorder=3)
        if label:
            labels[ax].append({"x": x, "y": p["value"], "text": p["label"], "kw": {"fontsize": 10}})

    def diamond(ax, p, x):
        markers[ax].append((x, p["value"], 620))
        ax.errorbar(x, p["value"], yerr=[[p["value"] - p["ci"][0]], [p["ci"][1] - p["value"]]],
                    fmt="none", ecolor=GOLD, alpha=0.6, capsize=4)
        ax.scatter(x, p["value"], s=620, marker="D", color=GOLD, edgecolor="black", lw=2, zorder=5)
        # gold labels go first so they get the best spot
        labels[ax].insert(0, {"x": x, "y": p["value"], "text": p["label"],
                              "kw": {"fontsize": 11, "fontweight": "bold"},
                              "offsets": [(0, 26), (-30, 0), (30, 0), (0, -30), (-30, 24), (30, 24)]})

    # ---- left: score vs size
    for p in base:
        circle(ax1, p, p["size_b"])
    for p in ft:
        b = by.get(p["base_system"])
        if b:
            ax1.annotate("", xy=(p["size_b"], p["value"]), xytext=(b["size_b"], b["value"]),
                         arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.8, shrinkA=12, shrinkB=16))
            gain = p["value"] - b["value"]
            labels[ax1].insert(0, {"x": p["size_b"], "y": (p["value"] + b["value"]) / 2,
                                   "text": f"{'+' if gain >= 0 else ''}{gain:.1f}{' pp' if metric == 'acc' else ''} from fine-tuning",
                                   "kw": {"fontsize": 10, "fontweight": "bold", "color": "#7E5109"},
                                   "offsets": [(-10, 0), (10, 0), (-10, 14), (10, 14), (-10, -14), (10, -14)]})
        diamond(ax1, p, p["size_b"])
    merged = {}
    for p in fr:
        merged.setdefault(round(p["value"], 1), []).append(p["label"])
    for i, (val, names) in enumerate(sorted(merged.items(), reverse=True)):
        c = FRONTIER_COLORS[i % len(FRONTIER_COLORS)]
        ax1.axhline(val, color=c, ls="--", lw=1.6, alpha=0.8)
        obstacles[ax1].append(ax1.text(0.01, val, f" frontier: {' = '.join(names)}: {val:.1f}", color=c,
                                       fontsize=10, va="bottom", transform=ax1.get_yaxis_transform()))
    sizes = [p["size_b"] for p in base + ft] or [1, 32]
    ax1.set_xscale("log")
    ax1.set_xlim(min(sizes) / 1.6, max(sizes) * 1.6)
    ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 32, 64) if min(sizes) / 1.6 <= t <= max(sizes) * 1.6]
    ax1.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax1.set_xlabel("Model size (B parameters, log scale)", fontsize=13, fontweight="bold")
    ax1.set_title("How small can it be? Fireworks-tunable models", fontsize=13)

    # ---- right: score vs cost (business value)
    costs = [p["cost_1k"] for p in points if p["cost_1k"] > 0]
    xmin, xmax = (min(costs) / 2.5, max(costs) * 2.5) if costs else (0.01, 100)
    if fr:
        ref = fr[0]
        ax2.add_patch(Rectangle((xmin, ref["value"]), ref["cost_1k"] - xmin, hi + 2 * pad - ref["value"],
                                color="#2ECC71", alpha=0.10, zorder=0))
        obstacles[ax2].append(ax2.text(xmin * 1.15, hi + 1.6 * pad, f"≥ {ref['label']} quality, cheaper",
                                       color="#1E8449", fontsize=10, va="top", fontweight="bold"))
    for p in base:  # names live on the left panel; colours + legend identify them here
        if p["cost_1k"] > 0:
            circle(ax2, p, p["cost_1k"], label=False)
    for p in ft:
        b = by.get(p["base_system"])
        if b and b["cost_1k"] > 0 and p["cost_1k"] > 0:
            ax2.annotate("", xy=(p["cost_1k"], p["value"]), xytext=(b["cost_1k"], b["value"]),
                         arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6, ls="--", shrinkA=12, shrinkB=16))
    for i, p in enumerate(fr):
        c = FRONTIER_COLORS[i % len(FRONTIER_COLORS)]
        ax2.scatter(p["cost_1k"], p["value"], s=420, marker="*", color=c, edgecolor="black", lw=1, zorder=4)
        markers[ax2].append((p["cost_1k"], p["value"], 420))
        labels[ax2].insert(0, {"x": p["cost_1k"], "y": p["value"], "text": f"{p['label']} (frontier)",
                               "kw": {"fontsize": 10, "color": c, "fontweight": "bold"}})
    summary = []
    best_ft = max(ft, key=lambda q: q["value"])["system"] if ft else None
    for p in ft:
        if p["cost_1k"] <= 0:
            continue
        diamond(ax2, p, p["cost_1k"])
        if fr and p["system"] == best_ft:
            ref = fr[0]
            mult = ref["cost_1k"] / p["cost_1k"]
            d = p["value"] - ref["value"]
            txt = (f"{mult:.1f}× cheaper than {ref['label']}\n"
                   f"{'+' if d >= 0 else ''}{d:.1f}{' pp' if metric == 'acc' else ' net'} vs frontier")
            labels[ax2].insert(1, {"x": p["cost_1k"], "y": p["value"], "text": txt,
                                   "kw": {"fontsize": 10.5, "fontweight": "bold", "color": "#7E5109",
                                          "bbox": dict(boxstyle="round,pad=0.35", fc="#FEF5E7", ec=GOLD),
                                          "arrowprops": dict(arrowstyle="-", color=GOLD, lw=1.2)},
                                   "offsets": [(-60, -60), (60, -60), (-150, -20), (-170, -55), (-120, 40), (-70, -90),
                                               (70, -90), (-60, -120), (60, -120), (-200, -100), (-220, 10)]})
            summary.append(f"{p['label']}: {p['value']:.1f} vs {ref['label']} {ref['value']:.1f} "
                           f"({'+' if d >= 0 else ''}{d:.1f}) at ${p['cost_1k']:.3f} vs ${ref['cost_1k']:.3f} "
                           f"per 1k questions ({mult:.1f}× cheaper)")
    ax2.set_xscale("log")
    ax2.set_xlim(xmin, xmax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=11, mfc=fcol[f], mec="#333", label=f"{f} (tunable base)")
               for f in families]
    handles += [Line2D([], [], marker="D", ls="", ms=12, mfc=GOLD, mec="black", label="fine-tuned (ours)"),
                Line2D([], [], marker="*", ls="", ms=16, mfc=FRONTIER_COLORS[0], mec="black", label="frontier API")]
    leg = ax2.legend(handles=handles, loc="lower right", fontsize=9.5, framealpha=0.9)
    obstacles[ax2].append(leg)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.2f}" if v < 1 else f"${v:,.0f}"))
    ax2.set_xlabel("Measured cost per 1,000 questions (USD, log scale)", fontsize=13, fontweight="bold")
    ax2.set_title("Business value: quality per dollar", fontsize=13)

    fig.suptitle(f"{'YKS STEM' if dataset == 'yks' else 'Olympiad'} — {split} split, n={n} identical questions",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005, "Error bars: 95% bootstrap CI. Cost = measured tokens × list price (serverless) or "
             "GPU-hours × measured wall time (dedicated / fine-tuned). Frontier sizes are not public.",
             ha="center", fontsize=9, color="#555")
    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=48, color="red", alpha=0.18, ha="center", va="center",
                 rotation=25, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ax in (ax1, ax2):  # after layout + limits are final, so pixel boxes are accurate
        place_labels(ax, labels[ax], markers[ax], obstacles[ax])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    return "\n".join(summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yks")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="acc", choices=["acc", "net40"])
    ap.add_argument("--frontier", nargs="+", default=["claude-opus-5.5", "gpt-6.1-sol"])
    ap.add_argument("--out")
    ap.add_argument("--systems", nargs="+", help="only plot these (frontier always included)")
    ap.add_argument("--demo", action="store_true", help="MOCK data layout preview (watermarked)")
    a = ap.parse_args()
    if a.demo:
        pts, n = demo_points()
        out = Path(a.out or "/tmp/finetune_value_DEMO.png")
        plot(pts, n, "acc", "yks", "test", out, watermark="MOCK DATA - LAYOUT PREVIEW")
    else:
        pts, n = collect(a.dataset, a.split, a.metric, a.frontier, a.systems)
        out = Path(a.out or ROOT / "results" / a.dataset / a.split / f"finetune_value_{a.metric}.png")
        s = plot(pts, n, a.metric, a.dataset, a.split, out)
        if s:
            print(s)
    print(f"saved {out}")


if __name__ == "__main__":
    main()


def plot_ablation(split: str = "test", out: Path | None = None) -> Path:
    """Base vs SFT v1 (targets without reasoning) vs SFT v2 (targets with reasoning traces).

        python -c "from yks.plot_finetune import plot_ablation; plot_ablation()"
    """
    rdir = ROOT / "results" / "yks" / split
    rows = {}
    for p in rdir.glob("*.jsonl"):
        if not p.name.endswith(".runs.jsonl"):
            rs = load_rows(p)
            rows[p.stem] = (100 * sum(r["correct"] for r in rs.values()) / len(rs), len(rs),
                            sum(r["out_tokens"] for r in rs.values()) / len(rs))
    sizes = [("Qwen3-0.6B", "qwen3-0.6b"), ("Qwen3-4B", "qwen3-4b"), ("Qwen3-8B", "qwen3-8b"), ("Qwen3-14B", "qwen3-14b")]
    variants = [("base", "{}", "#7F8C8D"), ("SFT v1: answers only", "sft-{}", "#E74C3C"),
                ("SFT v2: + reasoning traces", "sft2-{}", GOLD)]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    width = 0.27
    for vi, (vlabel, pat, color) in enumerate(variants):
        for si, (_, key) in enumerate(sizes):
            r = rows.get(pat.format(key))
            if not r:
                continue
            x = si + (vi - 1) * width
            ax.bar(x, r[0], width, color=color, edgecolor="#333", label=vlabel if si == 0 or vlabel not in
                   [h.get_label() for h in ax.patches] else None)
            ax.text(x, r[0] + 1, f"{r[0]:.0f}%\n{r[2]:.0f} tok", ha="center", va="bottom", fontsize=8.5)
    for name, c in (("claude-opus-5.5", "#7D3C98"),):
        if name in rows:
            ax.axhline(rows[name][0], color=c, ls="--", lw=1.4)
            ax.text(-0.45, rows[name][0] + 0.8, f"frontier ({name}): {rows[name][0]:.1f}%", color=c,
                    ha="left", fontsize=10)
    handles, labs = ax.get_legend_handles_labels()
    uniq = dict(zip(labs, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=10)
    ax.set_xticks(range(len(sizes)), [s for s, _ in sizes], fontsize=11)
    ax.set_ylabel("YKS accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", ls="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Fine-tuning ablation: distilling answers without reasoning removes the model's thinking\n"
                 "(same 517-575 synthetic, decontaminated questions; n=240 held-out; 'tok' = avg output tokens)",
                 fontsize=12)
    fig.tight_layout()
    out = out or rdir / "finetune_ablation.png"
    fig.savefig(out, dpi=160)
    return out
