# Frontier vs Laya vs Jev: Limitations and What Is Stopping Us

Single honest record. All numbers measured by us unless marked published.
No TDB roster locally (`/tmp/tdb` missing), so TDB numbers are historical.
JevBench numbers below are our internal harness (accuracy + mean proper
score), not the official chance-corrected composite — comparable across OUR
rows only.

## Measured standings (2026-10-01/02)

Full JevBench public, 231 items, `bb05a33` (`bench_full_231.json`):

| System | acc | proper | notes |
|---|---|---|---|
| Laya 421M (teacher) | 0.5801 | 73.35 | reference |
| Frontier v7 429M (full KD, 8.6k) | **0.5844** | **74.42** | +1 answer over Laya = noise |
| Frontier v8-ddp 429M (gold-first, 22k+1024) | 0.5541 | 68.45 | trails on hard |
| v7+v8 ensemble (avg probs) | 0.5411 | 72.54 | averaging the weaker drags; killed |

80-item fast subset (easy-heavy, flatters everyone): v7 0.80 > Laya 0.7875 =
v8-512 = v8-1024 > v8-ddp 0.775. Same one-answer noise band.

Public sets: AG News — v7 0.96 > Laya 0.95 > v8 0.945. Emotion — v8 **0.82** >
v3 0.767 > v7 0.693 > Laya 0.647 (our one genuine lead).

TDB v0.3 historical: Laya 62.78, v3 61.95, Jev 1.13 published 81.70 (unmeasured).

Verdict: **not frontier.** Parity with Laya on decisions, worse on hard,
better on emotion and on speed/size (v3: 168M, 26ms vs 421M, ~35ms).

## Limitation 1 — Hard reasoning capability (biggest)

Zero or near-zero families for ALL our models AND Laya: `trap 0.0`,
`ambiguous ~0.14`, `routing_hard 0.2-0.4`, `multi_hop 0.22-0.33`,
`probability 0.4-0.6`. More easy gold (AG/Emotion/synth) does not move these;
v8 proved it (22k items, JevBench down vs v7). Missing: multi-hop chaining,
numerical probability, policy-precedence routing under conflicting cues.

## Limitation 2 — Teacher ceiling shapes us

v7 (full Laya KD) is our best JevBench row: the student tracks the teacher.
Laya agrees with v9 gold only 83.5% (22845/27363); its 16.5% disagreement is
concentrated exactly where we need to beat it. Distilling harder cannot cross
a teacher that is wrong on the target rows — hence v9's hybrid (gold always,
KD 0.5 agree / 0.1 disagree) instead of pure KD.

## Limitation 3 — Long context: inference tricks exhausted

Audit (`audit_v8.jsonl`): multi_hop kept-token ratio 0.27 at 512, both models
0.278 — truncation artifact, not proof of anything. But: 1024 stage flat
(0.7875, proper 84.09 vs 83.69), evidence-v1 confidence-rank WORSE than
truncate (1/6 vs 2/6, walks into authored `surface_answer` distractors),
evidence-v2 (first+last+high-entropy) 4/6 on pilot but 0.222 vs 0.333 on full
multi_hop — small-sample luck, selection drops scattered facts. Chunk-dropping
of any ranking loses multi-hop evidence. Remaining path: long-context
TRAINING with answer signal, not selection at inference.

## Limitation 4 — Calibration is fragile

`temperature_by_options` is `{}` in every checkpoint (860–2900 cal items vs
thousands needed per bucket). Proper-score gaps (v8 83.77 vs Laya 85.01) come
substantially from this, not just argmax. Global temps only.

## Limitation 5 — Evaluation gaps

- 80-item subset overstated everything (easy-heavy); only full-231 counts.
- One-answer margins (v7 vs Laya, twice) are NOT leads; CIs overlap.
- v9 trains on disclosed JevBench-public gold, so post-v9 public numbers are
  public-trained numbers. Sealed/holdout honesty rests on the untouched
  final splits + AG/Emotion.
- Jev itself unmeasurable from here (closed API); 81.70 is a claim, not a target.

## Limitation 6 — Scale discipline

429M parity was reached, but 512→1024 and gold→KD→hybrid all land within
±0.02 of each other: supervision quality, not parameters or windows, binds.
Single T4-class hardware + ~30h/wk Kaggle quota caps run count; DDP 2×T4
works (26 min vs 45+, exact math) and is now the default vehicle.

## What v9 must do to change the verdict

Beat Laya on the hard families (`multi_hop`, `probability`, `routing_hard`,
`judge_hard`) on full-231 while holding emotion ≥ 0.80. Anything else —
dev accuracy, easy subsets, parameter counts — is not frontier evidence.
