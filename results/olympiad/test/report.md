# Results (olympiad / test, n=276 identical problem runs)

Baseline: `claude-opus-5.5`. Cost is estimated from measured token usage x list price in configs/systems.yaml. Latency is wall-clock per question at the run's concurrency. Olympiad: final answers graded with math-verify. With --repeats, each problem counts R times (CI is slightly optimistic).

| system | acc | 95% CI | Δacc vs base | wins/losses vs base | McNemar p | p50 s | p95 s | $/problem | cost vs base | avg out tok | escalated | parse fail | api err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oss120 | 72.5% | 67.0%–77.5% | -26.8 pp | 0/74 | 0.000 | 18.9 | 82.9 | $0.0039 | 0.11x | 6478 | 0% | 0 | 0 |
| gpt-6.1-sol | 97.8% | 96.0%–99.3% | -1.4 pp | 2/6 | 0.289 | 13.2 | 55.6 | $0.0122 | 0.34x | 1076 | 0% | 0 | 0 |
| claude-opus-5.5 | 99.3% | 98.2%–100.0% | +0.0 pp | 0/0 | 1.000 | 12.4 | 39.8 | $0.0364 | 1.00x | 1771 | 0% | 0 | 0 |

## Accuracy by competition

| system | aime_2025 | aime_2026 | brumo_2025 | cmimc_2025 | hmmt_feb_2025 | hmmt_feb_2026 | hmmt_nov_2025 | smt_2025 |
|---|---|---|---|---|---|---|---|---|
| oss120 | 70.0% | 76.7% | 90.0% | 57.5% | 70.0% | 66.7% | 70.0% | 79.2% |
| gpt-6.1-sol | 100.0% | 96.7% | 100.0% | 97.5% | 96.7% | 93.9% | 96.7% | 100.0% |
| claude-opus-5.5 | 100.0% | 96.7% | 100.0% | 100.0% | 100.0% | 97.0% | 100.0% | 100.0% |

## Accuracy by contest year (contamination check)

| system | 2025 | 2026 |
|---|---|---|
| oss120 | 72.8% (n=213) | 71.4% (n=63) |
| gpt-6.1-sol | 98.6% (n=213) | 95.2% (n=63) |
| claude-opus-5.5 | 100.0% (n=213) | 96.8% (n=63) |

## Accuracy by vote agreement (multi-sample systems)

| system | agreement | n | acc |
|---|---|---|---|
