# Results (olympiad / test, n=74 identical problem runs)

Baseline: `claude-opus-5.5`. Cost is estimated from measured token usage x list price in configs/systems.yaml. Latency is wall-clock per question at the run's concurrency. Olympiad: final answers graded with math-verify. With --repeats, each problem counts R times (CI is slightly optimistic).

| system | acc | 95% CI | Δacc vs base | wins/losses vs base | McNemar p | p50 s | p95 s | $/problem | cost vs base | avg out tok | escalated | parse fail | api err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oss120 | 74.3% | 63.5%–83.8% | -24.3 pp | 0/18 | 0.000 | 14.2 | 69.1 | $0.0033 | 0.10x | 5541 | 0% | 0 | 0 |
| gpt-6.1-sol | 95.9% | 90.5%–100.0% | -2.7 pp | 1/3 | 0.625 | 12.0 | 32.6 | $0.0102 | 0.30x | 893 | 0% | 0 | 0 |
| qwen3-14b | 70.3% | 59.5%–81.1% | -28.4 pp | 0/21 | 0.000 | 327.1 | 878.9 | $0.0194 | 0.58x | 14246 | 0% | 2 | 0 |
| claude-opus-5.5 | 98.6% | 95.9%–100.0% | +0.0 pp | 0/0 | 1.000 | 11.6 | 39.8 | $0.0336 | 1.00x | 1632 | 0% | 0 | 0 |

Cost notes:
- `qwen3-14b`: dedicated deployment at $16.00/hr, measured 825 questions/hr at concurrency 32 -> GPU-hour cost included

## Accuracy by competition

| system | aime_2025 | aime_2026 | brumo_2025 | cmimc_2025 | hmmt_feb_2025 | hmmt_feb_2026 | hmmt_nov_2025 | smt_2025 |
|---|---|---|---|---|---|---|---|---|
| oss120 | 70.0% | 20.0% | 100.0% | 81.8% | 100.0% | 62.5% | 63.6% | 78.6% |
| gpt-6.1-sol | 100.0% | 80.0% | 100.0% | 100.0% | 87.5% | 87.5% | 100.0% | 100.0% |
| qwen3-14b | 90.0% | 20.0% | 100.0% | 63.6% | 87.5% | 50.0% | 63.6% | 71.4% |
| claude-opus-5.5 | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Accuracy by contest year (contamination check)

| system | 2025 | 2026 |
|---|---|---|
| oss120 | 80.3% (n=61) | 46.2% (n=13) |
| gpt-6.1-sol | 98.4% (n=61) | 84.6% (n=13) |
| qwen3-14b | 77.0% (n=61) | 38.5% (n=13) |
| claude-opus-5.5 | 100.0% (n=61) | 92.3% (n=13) |

## Accuracy by vote agreement (multi-sample systems)

| system | agreement | n | acc |
|---|---|---|---|
