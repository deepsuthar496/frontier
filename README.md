# Frontier Decision Engine — fast, local, typed decisions

Flagship: **v8** (`frontier_ckpt_v8_ddp`, 429M, 1024 ctx) — best clean-room
generalization. Reference: **v7** (best clean public-bench row). History and
per-run records live in `JEV_BOTTLENECKS.md`; the remote-GPU runbook in
`KAGGLE_WORKFLOW.md`.

## Measured standings (our harness unless noted)

Full JevBench public, 231 items, accuracy + mean proper score
(`bench_full_231.json`, rev `bb05a33`):

| System | acc | proper | Params |
|---|---|---|---|
| Laya english | 0.5801 | 73.35 | 421M |
| Frontier v7 | 0.5844 | 74.42 | 429M |
| Frontier v8-ddp | 0.5541 | 68.45 | 429M |
| v7+v8 ensemble | 0.5411 | 72.54 | — |

v8 leads sealed/emotion (below) while trailing full-231 public: it trades
public-hard memorization for clean-room generalization. v9 scores 0.827
public but trained on those items (disclosed) — excluded from ranking.

Sealed-60 (fresh templates, unseen by all training, `sealed_eval.json`):

| System | acc | proper |
|---|---|---|
| **v8** | **0.85** | **85.98** |
| v7 | 0.7667 | 83.91 |
| Laya english | 0.75 | 82.72 |
| v9 | 0.6667 | 83.18 |

Public sets (`eval_public_v8ddp.json`): AG News — v7 0.96 / v8 0.945 /
Laya 0.95. Emotion — **v8 0.82** / v3 0.767 / v7 0.693 / Laya 0.647.

Locked 708-item holdout, never trained on (`final_eval_v9v7.json`):
v9 0.901 / Laya 0.573 / v7 0.504 (v9 rows are same-distribution shards).

Official TDB v0.3 (historical, roster currently unavailable):
Jev 1.13 published 81.70 · Laya 62.78 · v3 61.95 · v4 61.17* · v5 60.99*.
Halves reproduce full scores ±0.12.

## Latency (measured)

| System | 1q short | 1q long | GPU 1q |
|---|---|---|---|
| Laya 421M | 222ms CPU / ~31ms T4 | 941ms CPU | ~31–44ms |
| **v3 168M (fast low-end)** | **85ms CPU** | **356ms CPU** | ~26ms |
| v8/v9 429M | ~230ms CPU | ~950ms CPU | ~36ms |

Speed story split: v3 is the fast low-end model (2.6× Laya on CPU);
v8/v9 buy accuracy at Laya-class speed. Batched GPU: encoder single-pass
does ~1100 dec/s at B64 vs ~87 for autoregressive 12B-class (Winnow tables).

## How it is built

- Backbone: `ModernBERT-large` 395M (hidden 1024, 28 layers, 8k RoPE) +
  2-layer refine + **SchemaCrossAttentionHead** (option markers as queries
  cross-attend the doc — same single-pass marker scheme as Laya, Jev-compatible).
- Loss: `CE(gold) + 0.1·Brier + 0.1·RPS(ordinal) + α·τ²·KL(teacher‖student)`,
  `α`: agree 0.5 / disagree 0.1 (`common.KD_POLICY`, single source of truth).
- Teachers: Laya (v6/v7), Laya + gold hybrid (v8), + disclosed JevBench-public
  gold (v9). Kev-4B capture in progress for v10.
- Splits: train/dev/cal/**final untouched**, sharded by group with class
  coverage asserted. Job: `data_v9.py` → label → `train_v9_ddp.py` (DDP 2×T4).
- E0 gates: canonical option mapping + round-trip tests (`tests/`),
  teacher id/hash validation, KD policy persisted per checkpoint.

## Honest limits

- No sealed-external benchmark: strongest evidence is the sealed-60 and the
  locked-708, both constructed in-house. Jev 81.70 is a published claim.
- v9's public numbers are public-trained numbers (disclosed in
  `data_v9.py`); its sealed-60 (0.667) shows template overfit — v9 is a
  specialist, not the flagship.
- `temperature_by_options` is `{}` everywhere (cal splits too small per
  bucket); proper-score gaps are partly calibration, not just accuracy.
- `act_probability` is an untrained signal — gate on `confidence`.
- English-only.

## Use

```python
from frontier.agent import FrontierAgent
a = FrontierAgent("frontier_ckpt_v8_ddp")  # flagship; v7 for the public-bench reference
r = a.system_one("Hi, we were billed twice...", {
  "department": {"type": "choice", "instructions": "Which department?",
                 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "else"}},
  "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"}})
```

Long states: `a.cfg["evidence"] = {"enable": True, "strategy": "v2"}` (experimental;
helps some longs, hurts multi-hop — see evals).
