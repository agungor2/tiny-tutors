# Training a small tutor model

This is the recipe used in the experiments: distill verified frontier reasoning into a small open model
with managed LoRA SFT on Fireworks, then evaluate it on the same held-out questions as every other system.

## 1. Pick a base model

```bash
python -m yks.list_models tunable            # every SFT/RL-tunable model, sorted by size
python -m yks.list_models tunable qwen llama
```

Prefer a small **dense** model with ≥16K training context. We used the Qwen3 ladder (0.6B, 4B, 8B, 14B),
plus Nemotron Lightning 30B-A3B as an MoE comparison.

## 2. Build training data that never touches the evaluation set

```bash
python -m yks.make_sft --n 600 --generator gpt-6.1-sol --verifier claude-opus-5.5   # questions + keys
python -m yks.make_sft_reasoning --tries 2                                           # reasoning targets
```

- **Synthetic questions:** written to the YKS topic taxonomy (`yks/topics.py`). The public TurkishMMLU
  questions are only ever used for evaluation.
- **Cross-model verification:** one model writes the question and its key, and a different model solves
  it independently. A question is kept only if both agree.
- **Decontamination:** anything sharing a word 8-gram, or with >50% word-3-gram Jaccard, with any
  tune/test question is dropped.
- **Targets carry the reasoning:** each target has the teacher's reasoning in `reasoning_content` (which
  Fireworks trains as the assistant's thinking) plus a final answer ending in `Cevap: X`. A target is
  kept only if the teacher reaches the verified key (rejection sampling).

The counts, rules and generation cost are written to `data/sft*/summary.json`.

> **Why the reasoning matters.** Our first version trained on final solutions only. Every model learned
> to stop reasoning (~1,600 → ~400 output tokens) and lost 13–25 points of accuracy. Keeping the
> reasoning traces recovered most of it. See the ablation in the README.

## 3. Train

```bash
firectl dataset create yks-sft2-train data/sft_v2/train.jsonl
firectl dataset create yks-sft2-val   data/sft_v2/val.jsonl
firectl supervised-fine-tuning-job create \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset yks-sft2-train --evaluation-dataset yks-sft2-val \
  --output-model yks-sft2-qwen3-8b --max-context-length 16384
```

We used the defaults (LoRA rank 8). Training on ~530 examples took about 8–10 minutes per model.
`yks/submit_sft.py` automates the upload and job creation.

## 4. Deploy, evaluate, delete

Trained models are served on dedicated deployments. `yks/deploy_sft.py` waits for the job, deploys on
the cheapest working shape, runs the evaluation at concurrency 32 and always deletes the deployment:

```bash
export FIREWORKS_ACCOUNT_ID=<your-account>
python -m yks.deploy_sft --system sft2-qwen3-8b --job-file results/sft_jobs/sft2-qwen3-8b.json
python -m yks.report --split test
```

**GPU-hour costing:** for dedicated models, cost per question = GPU $/hr × measured eval wall time ÷
questions. That depends on throughput, so compare dedicated models at the same concurrency.
Deployment spin-up (~5 min) is excluded from per-question cost.

## 5. Charts

```bash
python -m yks.plot_finetune --split test --frontier claude-opus-5.5 gpt-6.1-sol \
  --systems qwen3-0.6b qwen3-4b qwen3-8b qwen3-14b sft2-qwen3-4b sft2-qwen3-8b sft2-qwen3-14b \
  --out results/yks/test/finetune_value.png
python -c "from yks.plot_finetune import plot_ablation; plot_ablation()"
```

- **Value chart:** score vs model size (left) and vs measured cost (right). Base models are circles,
  fine-tuned models are diamonds with an arrow from their base, and frontier models are dashed lines.
- **Ablation chart:** base vs answers-only SFT vs SFT with reasoning traces, per model size.

## Reading the results

- `sft2-qwen3-8b` vs `qwen3-8b` (same base, same questions) is the **fine-tuning delta**.
- Any fine-tuned model vs `claude-opus-5.5` / `gpt-6.1-sol` is the **distance to the frontier**.
