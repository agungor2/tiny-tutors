# Results (yks / test, n=240 identical questions)

Baseline: `claude-opus-5.5`. Cost is estimated from measured token usage x list price in configs/systems.yaml. Latency is wall-clock per question at the run's concurrency. net/40 = YKS net (correct - wrong/4) scaled to a 40-question test; blanks only occur in exam-mode systems.

| system | acc | 95% CI | net/40 | blank | Δacc vs base | wins/losses vs base | McNemar p | p50 s | p95 s | $/40-q exam | cost vs base | avg out tok | escalated | parse fail | api err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oss120 | 88.3% | 84.2%–92.1% | 34.2 | 0% | -6.2 pp | 4/19 | 0.003 | 3.0 | 6.5 | $0.0191 | 0.04x | 752 | 0% | 2 | 0 |
| nemotron-lightning | 61.3% | 55.0%–67.5% | 20.6 | 0% | -33.3 pp | 2/82 | 0.000 | 7.3 | 202.8 | $0.0270 | 0.06x | 3345 | 0% | 56 | 0 |
| cascade-nemo-120 | 91.7% | 87.9%–95.0% | 35.8 | 0% | -2.9 pp | 3/10 | 0.092 | 49.5 | 563.5 | $0.1087 | 0.23x | 10826 | 58% | 1 | 5 |
| gpt-6.1-sol | 94.6% | 91.7%–97.1% | 37.3 | 0% | +0.0 pp | 3/3 | 1.000 | 3.9 | 8.2 | $0.1297 | 0.27x | 252 | 0% | 4 | 0 |
| claude-opus-5.5 | 94.6% | 91.7%–97.1% | 37.3 | 0% | +0.0 pp | 0/0 | 1.000 | 5.7 | 11.4 | $0.4793 | 1.00x | 534 | 0% | 0 | 0 |

## Accuracy by subject

| system | Chemistry | Mathematics | Physics |
|---|---|---|---|
| oss120 | 88.8% | 93.8% | 82.5% |
| nemotron-lightning | 62.5% | 65.0% | 56.2% |
| cascade-nemo-120 | 90.0% | 92.5% | 92.5% |
| gpt-6.1-sol | 95.0% | 95.0% | 93.8% |
| claude-opus-5.5 | 93.8% | 93.8% | 96.2% |

## Accuracy by vote agreement (multi-sample systems)

| system | agreement | n | acc |
|---|---|---|---|
| cascade-nemo-120 | 100% | 101 | 95.0% |
| cascade-nemo-120 | 67% | 83 | 91.6% |
| cascade-nemo-120 | 33% | 44 | 86.4% |
| cascade-nemo-120 | 0% | 12 | 83.3% |
