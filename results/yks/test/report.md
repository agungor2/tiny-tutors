# Results (yks / test, n=240 identical questions)

Baseline: `claude-opus-5.5`. Cost is estimated from measured token usage x list price in configs/systems.yaml. Latency is wall-clock per question at the run's concurrency. net/40 = YKS net (correct - wrong/4) scaled to a 40-question test; blanks only occur in exam-mode systems.

| system | acc | 95% CI | net/40 | blank | Δacc vs base | wins/losses vs base | McNemar p | p50 s | p95 s | $/40-q exam | cost vs base | avg out tok | escalated | parse fail | api err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oss120 | 88.3% | 84.2%–92.1% | 34.2 | 0% | -6.2 pp | 4/19 | 0.003 | 3.0 | 6.5 | $0.0191 | 0.04x | 752 | 0% | 2 | 0 |
| sft-qwen3-14b | 71.7% | 65.8%–77.1% | 25.8 | 0% | -22.9 pp | 2/57 | 0.000 | 3.1 | 5.6 | $0.0208 | 0.04x | 337 | 0% | 2 | 0 |
| nemotron-lightning | 61.3% | 55.0%–67.5% | 20.6 | 0% | -33.3 pp | 2/82 | 0.000 | 7.3 | 202.8 | $0.0270 | 0.06x | 3345 | 0% | 56 | 0 |
| sft-qwen3-4b | 58.8% | 52.5%–65.0% | 19.4 | 0% | -35.8 pp | 1/87 | 0.000 | 1.6 | 3.4 | $0.0357 | 0.07x | 405 | 0% | 3 | 0 |
| sft2-qwen3-4b | 80.0% | 75.0%–85.0% | 30.0 | 0% | -14.6 pp | 2/37 | 0.000 | 2.7 | 9.1 | $0.0419 | 0.09x | 789 | 0% | 1 | 0 |
| qwen3-4b | 83.8% | 78.8%–88.3% | 31.9 | 0% | -10.8 pp | 2/28 | 0.000 | 4.4 | 18.8 | $0.0449 | 0.09x | 1436 | 0% | 0 | 0 |
| oss120-high | 87.9% | 84.2%–91.7% | 34.0 | 0% | -6.7 pp | 4/20 | 0.002 | 7.6 | 36.2 | $0.0455 | 0.09x | 1859 | 0% | 15 | 5 |
| qwen3-0.6b | 47.9% | 41.7%–54.2% | 14.0 | 0% | -46.7 pp | 5/117 | 0.000 | 7.2 | 62.8 | $0.0609 | 0.13x | 1740 | 0% | 6 | 0 |
| sft-qwen3-0.6b | 27.5% | 22.1%–33.3% | 3.8 | 0% | -67.1 pp | 4/165 | 0.000 | 2.7 | 90.0 | $0.0612 | 0.13x | 1277 | 0% | 11 | 0 |
| sft-qwen3-8b | 74.2% | 68.8%–79.6% | 27.1 | 0% | -20.4 pp | 2/51 | 0.000 | 4.7 | 50.4 | $0.0626 | 0.13x | 422 | 0% | 1 | 0 |
| qwen3-8b | 87.1% | 82.9%–91.2% | 33.5 | 0% | -7.5 pp | 2/20 | 0.000 | 13.9 | 53.1 | $0.0628 | 0.13x | 1651 | 0% | 1 | 0 |
| sft2-qwen3-8b | 84.6% | 80.0%–88.8% | 32.3 | 0% | -10.0 pp | 4/28 | 0.000 | 10.7 | 55.0 | $0.0726 | 0.15x | 842 | 0% | 4 | 0 |
| cascade-nemo-120 | 91.7% | 87.9%–95.0% | 35.8 | 0% | -2.9 pp | 3/10 | 0.092 | 49.5 | 563.5 | $0.1087 | 0.23x | 10826 | 58% | 1 | 5 |
| yks-sft | 7.9% | 4.6%–11.7% | -6.0 | 0% | -86.7 pp | 0/208 | 0.000 | 28.9 | 35.0 | $0.1138 | 0.24x | 5761 | 0% | 180 | 0 |
| qwen3-14b | 88.8% | 84.6%–92.5% | 34.4 | 0% | -5.8 pp | 1/15 | 0.001 | 11.5 | 45.3 | $0.1267 | 0.26x | 1238 | 0% | 0 | 0 |
| oss120-sc8 | 88.3% | 84.2%–92.1% | 34.2 | 0% | -6.2 pp | 4/19 | 0.003 | 10.3 | 34.0 | $0.1273 | 0.27x | 5129 | 0% | 1 | 60 |
| gpt-6.1-sol | 94.6% | 91.7%–97.1% | 37.3 | 0% | +0.0 pp | 3/3 | 1.000 | 3.9 | 8.2 | $0.1297 | 0.27x | 252 | 0% | 4 | 0 |
| sft2-qwen3-14b | 85.8% | 81.2%–90.0% | 32.9 | 0% | -8.8 pp | 2/23 | 0.000 | 5.9 | 16.5 | $0.1552 | 0.32x | 908 | 0% | 3 | 0 |
| claude-opus-5.5-exam | 93.3% | 90.0%–96.2% | 37.0 | 3% | -1.2 pp | 0/3 | 0.250 | 5.2 | 10.1 | $0.4503 | 0.94x | 476 | 0% | 0 | 0 |
| claude-opus-5.5 | 94.6% | 91.7%–97.1% | 37.3 | 0% | +0.0 pp | 0/0 | 1.000 | 5.7 | 11.4 | $0.4793 | 1.00x | 534 | 0% | 0 | 0 |
| claude-opus-5.5-high | 94.6% | 91.7%–97.1% | 37.3 | 0% | +0.0 pp | 1/1 | 1.000 | 5.7 | 11.0 | $0.4851 | 1.01x | 542 | 0% | 0 | 0 |

Cost notes:
- `qwen3-0.6b`: dedicated deployment at $8.00/hr, measured 5,252 questions/hr at concurrency 32 -> GPU-hour cost included
- `qwen3-14b`: dedicated deployment at $16.00/hr, measured 5,051 questions/hr at concurrency 32 -> GPU-hour cost included
- `qwen3-4b`: dedicated deployment at $13.00/hr, measured 11,585 questions/hr at concurrency 32 -> GPU-hour cost included
- `qwen3-8b`: dedicated deployment at $8.00/hr, measured 5,099 questions/hr at concurrency 32 -> GPU-hour cost included
- `sft-qwen3-0.6b`: dedicated deployment at $8.00/hr, measured 5,227 questions/hr at concurrency 32 -> GPU-hour cost included
- `sft-qwen3-14b`: dedicated deployment at $16.00/hr, measured 30,777 questions/hr at concurrency 32 -> GPU-hour cost included
- `sft-qwen3-4b`: dedicated deployment at $13.00/hr, measured 14,583 questions/hr at concurrency 32 -> GPU-hour cost included
- `sft-qwen3-8b`: dedicated deployment at $8.00/hr, measured 5,111 questions/hr at concurrency 32 -> GPU-hour cost included
- `sft2-qwen3-14b`: dedicated deployment at $16.00/hr, measured 4,123 questions/hr at concurrency 32 -> GPU-hour cost included
- `sft2-qwen3-4b`: dedicated deployment at $13.00/hr, measured 12,403 questions/hr at concurrency 32 -> GPU-hour cost included
- `sft2-qwen3-8b`: dedicated deployment at $8.00/hr, measured 4,406 questions/hr at concurrency 32 -> GPU-hour cost included
- `yks-sft`: dedicated deployment at $13.00/hr, measured 4,570 questions/hr at concurrency 32 -> GPU-hour cost included

## Accuracy by subject

| system | Chemistry | Mathematics | Physics |
|---|---|---|---|
| oss120 | 88.8% | 93.8% | 82.5% |
| sft-qwen3-14b | 66.2% | 75.0% | 73.8% |
| nemotron-lightning | 62.5% | 65.0% | 56.2% |
| sft-qwen3-4b | 62.5% | 58.8% | 55.0% |
| sft2-qwen3-4b | 76.2% | 88.8% | 75.0% |
| qwen3-4b | 85.0% | 90.0% | 76.2% |
| oss120-high | 90.0% | 91.2% | 82.5% |
| qwen3-0.6b | 41.2% | 68.8% | 33.8% |
| sft-qwen3-0.6b | 27.5% | 28.7% | 26.2% |
| sft-qwen3-8b | 75.0% | 73.8% | 73.8% |
| qwen3-8b | 83.8% | 91.2% | 86.2% |
| sft2-qwen3-8b | 81.2% | 88.8% | 83.8% |
| cascade-nemo-120 | 90.0% | 92.5% | 92.5% |
| yks-sft | 10.0% | 5.0% | 8.8% |
| qwen3-14b | 87.5% | 92.5% | 86.2% |
| oss120-sc8 | 87.5% | 93.8% | 83.8% |
| gpt-6.1-sol | 95.0% | 95.0% | 93.8% |
| sft2-qwen3-14b | 85.0% | 88.8% | 83.8% |
| claude-opus-5.5-exam | 92.5% | 92.5% | 95.0% |
| claude-opus-5.5 | 93.8% | 93.8% | 96.2% |
| claude-opus-5.5-high | 93.8% | 93.8% | 96.2% |

## Accuracy by vote agreement (multi-sample systems)

| system | agreement | n | acc |
|---|---|---|---|
| cascade-nemo-120 | 100% | 101 | 95.0% |
| cascade-nemo-120 | 67% | 83 | 91.6% |
| cascade-nemo-120 | 33% | 44 | 86.4% |
| cascade-nemo-120 | 0% | 12 | 83.3% |
| oss120-sc8 | 100% | 149 | 95.3% |
| oss120-sc8 | 88% | 14 | 78.6% |
| oss120-sc8 | 75% | 21 | 85.7% |
| oss120-sc8 | 62% | 27 | 88.9% |
| oss120-sc8 | 50% | 4 | 75.0% |
| oss120-sc8 | 38% | 13 | 53.8% |
| oss120-sc8 | 25% | 8 | 50.0% |
| oss120-sc8 | 12% | 3 | 100.0% |
| oss120-sc8 | 0% | 1 | 0.0% |
