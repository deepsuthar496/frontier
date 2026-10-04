# Frontier Bottlenecks (single record): Accuracy + Performance

All numbers measured by us unless marked published. No TDB roster locally, so
TDB numbers are historical. JevBench numbers are our internal harness
(accuracy + mean proper score), comparable across OUR rows only.

## 0. Performance first (the primary goal)

Measured per-question latency (CPU = low-end proxy; T4 fp16 otherwise):

| System | CPU short | CPU long | T4 1q | RAM fp16 |
|---|---|---|---|---|
| Laya 421M | 222ms | 941ms | ~31–44ms | ~800MB |
| v3 168M | **85ms** | **356ms** | ~26ms | ~340MB |
| v8/v9/v10 429M | ~230ms | ~950ms | ~36ms | ~860MB |

Findings:
- The "faster on low-end" claim belongs to **v3 only** (2.6× Laya on CPU,
  sub-100ms short). The 429M line bought accuracy at Laya's speed — it never
  was low-end, and shipping v9/v10 as the "fast" model would be false.
- Low-end target shape: **v3-int8** (~170MB, Winnow's Q8 precedent: −0.4pp
  cost). Not yet built — first performance work item (P1 below).
- GPU batching is our structural moat (encoder single-pass ~1100 dec/s at B64
  vs ~87 for 12B autoregressive, per Winnow's own tables). Never trade it away
  for autoregressive reasoning tricks.
- v10 public eval: 0.8615/89.69 (public-trained, disclosed); latency of v10 == v8 class.

### Performance work items

- **P1 — v3-int8 low-end artifact.** Quantize the 168M line (ONNX,
  `export_onnx.py` exists), verify ≤0.5pp drift on sealed-60 + emotion,
  publish size/latency/RAM table. This is the shippable fast model.
- **P2 — Batched serving path.** Shared-state multi-question in one forward
  with prefix-counted usage accounting (Winnow counts the shared prefix once;
  adopt). Measure B1/B8/B32 latency + dec/s on T4 like their tables.
- **P3 — Unpadded/bucketed inference.** Variable-length batching without pad
  waste (ModernBERT supports unpadding; needs flash-attn-2 path check on T4).
- **P4 — fp16-first artifacts.** Every release ships fp16 (half bytes, Laya
  parity at 804MB); fp32 stays training-only. Already done for v8+.

## 1. Measured standings

Full JevBench public, 231 items, `bb05a33` (`bench_full_231.json`):

| System | acc | proper | notes |
|---|---|---|---|
| Laya 421M (teacher) | 0.5801 | 73.35 | reference |
| Frontier v7 429M (full KD, 8.6k) | **0.5844** | **74.42** | +1 answer = noise |
| Frontier v8-ddp 429M (gold-first, 22k+1024) | 0.5541 | 68.45 | trails on hard |
| v7+v8 ensemble (avg probs) | 0.5411 | 72.54 | averaging weaker drags; killed |
| Frontier v9 (public-trained, disclosed) | 0.8268 | 87.72 | contaminated; excluded |
| Frontier v10 (public-trained, disclosed) | 0.8615 | 89.69 | contaminated; excluded |

Sealed-60, fresh templates unseen by all training (`sealed_eval.json`):

| System | acc | proper |
|---|---|---|
| **v8 / v10** | **0.85** | ~85 |
| v7 | 0.7667 | 83.91 |
| Laya english | 0.75 | 82.72 |
| v9 | 0.6667 | 83.18 |

Public sets: AG — v7 0.96 / v8 0.945 / Laya 0.95. Emotion — **v8 0.82** /
v3 0.767 / v7 0.693 / Laya 0.647 (genuine lead).

Locked-708 holdout, never trained on: v10 0.908 / v9 0.901 / Laya 0.573 /
v7 0.504 (same-distribution shards — strong but not sealed-external).

TDB v0.3 historical: Jev 1.13 published 81.70 · Laya 62.78 · v3 61.95.

Verdict: **not frontier overall.** Parity on decisions, worse on hard,
better on emotion and efficiency. v9/v10 public rows are contaminated.

## 2. Accuracy bottlenecks (research list, ranked)

### A1. Template overfit (proven by v9 sealed 0.667 vs v8 0.85)

Training on JevBench-public gold + synthetic templates teaches phrasing, not
reasoning. v9 memorized public-231 (0.827) and collapsed on new templates.
Research: verifiable-reward supervision (programmatic labels, RLVR-style),
real task rows (the 21 TDB source sets), never more generators of the same
shape. Counterfactual pairs (same stem, flipped decisive slot, same partition)
as the unit of data, not rows.

### A2. No joint option reasoning

One independent logit per option marker. Anything like "A only if B holds",
pairwise exclusion, or precedence under conflict (the near-zero rows in
`judge_hard`/`routing_hard`) needs listwise comparison — a cross-option head
over the K option states. Architecture gap, not a data gap.

### A3. No scratchpad / sequential inference

One encoder pass, no intermediate steps. Multi-hop needs chaining; the
non-autoregressive answers are recurrent depth (re-entrant refinement) or
distilled intermediate heads (evidence / rule / prerequisite / exception as
training-only targets — plan E6, untouched). Never chain-of-thought text at
inference: it destroys the batching moat (P-section).

### A4. Teacher ceiling is now Kev-4B (71.4)

Laya (58.4) is exhausted as a teacher. Kev-4B capture done (agree 22932 /
disagree 3516, 8k context so longs were seen). 85+ teachers (Winnow/Jev) need
>T4 or closed APIs. Research: teacher ensembles, and per-row diagnosis of
exactly which hard rows Kev gets wrong (target those with gold, not more KD).

### A5. Calibration gap (proper score, not argmax)

Hard Brier ~0.35 vs Jev's 0.18. Buckets are empty (`temperature_by_options`
`{}` in every checkpoint — cal splits too small per bucket). Research:
regularized grouped temperatures (shrinkage to global, plan's formula),
bigger cal splits, reliability plots per family. Note: temperature never
changes argmax — this buys proper score only.

### A6. Preference layer missing (RLCD/DPO/GRPO)

Everyone above us trains SFT-then-preference (Laya RLCD; surogate documents
GRPO/DPO tooling). We stop at SFT+KD. Research: DPO on chosen/rejected
decision pairs (hard rows we get wrong vs gold), then GRPO with proper-score
reward. Untouched.

### A7. Long context: training, not selection (settled)

94% truncation on multi_hop at 512; 1024 stage flat (0.7875); evidence-v1
worse than truncate (walks into authored distractors); evidence-v2 4/6 pilot
then 0.222 vs 0.333 full multi_hop (small-sample luck; dropping chunks loses
scattered facts). Chunk-dropping of any ranking is dead. Remaining: answer
signal over full states (2–3k tokens, fits T4 with grad-ckpt), or hierarchical
all-chunk encoder (keeps every representation — different from ranking).

## 3. What v10 proved and what it didn't

- Hybrid (gold + agree-0.5/disagree-0.1 Kev KD) from a clean base: sealed tie
  with v8 (0.85), public jump (0.8615), locked-final lead (0.908). Stronger
  teacher + clean lineage works; public ingestion is what poisoned v9.
- Not proven: sealed-external generalization (all sealed sets are in-house),
  TDB/Jev comparability, low-end latency of anything but v3.

## 5. v11 results (2026-10-03/04): what moved and what didn't

v11-base (v10 + 8k solver-verified hard rows, Kev KD, DDP):
dev 0.947 · hard-dev 1.000 · sealed-60 **0.9167** · AG **0.97** · emotion 0.7867.
v11-gold-hard (KD off on hard rows): dev 0.944 · hard-dev 1.000 ·
sealed-60 **0.9333** · AG 0.97 · emotion 0.78.
Hard-sealed templates (812 unseen parameterizations): **0.27 both variants**.

Readings:
- KD policy on hard rows is irrelevant (all variant deltas ≤ 1 answer).
  Selective variant retired unrun (bounded between identical endpoints).
- Sealed-60 gain (+6.7 over v10) is real transfer to new domains.
- Hard-sealed collapse is template memorization: same generator, new numbers
  fail while seen templates score 1.000. Both v10 and v11 fail identically,
  so the teacher is exonerated — the generator's parameter coverage is guilty.
- Emotion dip (0.82→0.78) comes from the hard-data distribution shift itself,
  identical in both variants — not from KD.

### A8. Template diversity (new #1 for v12)

Generators must vary the decisive parameters, not just names: score cutoffs,
rates, thresholds, rubric combinations — with counterfactual pairs at every
setting. Measure unseen-parameterization accuracy during development
(hard-sealed style), never just unseen-instance accuracy (hard-dev lies:
1.000 vs 0.27 on the same generator).
### A9. Intermediate supervision E6 (new for v12)

Evidence/rule/prerequisite/exception heads as training-only targets, one at
a time. The solver blocks already emit intermediates — wire one into the loss
and test whether the model computes instead of memorizes.

## 6. Acceptance bar (unchanged)

Hard-family wins over Laya on unseen templates while holding emotion ≥ 0.80,
with CIs and item counts — plus, now, a latency/RAM table for the shipped
artifact (P1). Dev accuracy, easy subsets, and parameter counts are not
evidence.
