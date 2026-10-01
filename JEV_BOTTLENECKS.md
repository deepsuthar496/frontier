# Bottlenecks to Jev Performance

Measured context (this T4, 2026-10-01):

| System | Params | JevBench-fast (80 items) acc | proper | AG News (200) | Emotion (150) | 1q latency |
|---|---|---|---|---|---|---|
| Laya english | 421M | 0.7875 | 85.01 | 0.95 | 0.647 | ~31–44ms |
| Frontier v3 (TDB champion) | 168M | 0.5625 | 69.19 | 0.905 | 0.767 | 26ms |
| Frontier v7 (large) | 429M | **0.80** | **85.92** | **0.96** | 0.693 | 36.5ms |
| Jev 1.13 (published, unmeasured here) | closed | 63.29 board / 74.4 v1.2 | — | — | — | API |

TDB roster (`/tmp/tdb`) is missing, so v7 has no official TDB number. v3 holds
TDB `61.95` full / `61.83` half vs Laya `62.78` / `62.90`. Jev `81.70` is a
published claim; it cannot be verified from here.

## 1. Teacher ceiling (biggest)

v7 distills `convaiinnovations/laya` soft logits (`teacher_labels.npz`, 8600
items) with `CE + tau^2·KL + Brier + RPS` (`common.py:90`, `train_v7_large.py`).
A distill student tracks its teacher: v7 beats Laya on this subset (+0.0125 acc,
+0.91 proper) but Laya-to-Jev is still ~15 points of unknown territory.
There is no RLCD (REINFORCE + group baseline), no TD-λ episode objective —
SFT-distill only.

Unlock: real task labels from a stronger-than-Laya signal, then RLCD on top.

## 2. Data is not the real distribution

Training is synth + AG-style teacher items. All three systems score `0.0` on
`multi_hop`, `probability`, `routing_hard` — missing capability, not noise.
v7 trails v3 on Emotion (`0.693` vs `0.767`): distilling Laya (`0.647`) dragged
a strength down. TDB contamination rule forbids training on TDB items.

Unlock: teacher-label the 21 real TDB source sets (CFPB, HotpotQA, FiNER, …)
instead of inventing lookalikes (`BLOCKERS.md:17-24`).

## 3. Long context untrained

Train/infer at `max_len 512 / head_max_len 192`; hard longs run ~3746 tokens
and truncate. `long_policy 0.5` (n=2) is luck, not coverage.

Unlock: train at 1024–2048 on real long states + trajectory objectives.

## 4. Calibration sample starvation

860 val items → `temperature_by_options: {}` (empty in
`frontier_ckpt_v7/frontier_config.json`). Buckets need 2000+ samples each;
ours fits global temps only. ECE looks fine, fragilly.

Unlock: calibrate on thousands of teacher-labeled items.

## 5. Compute budget

2 epochs, one T4 (~35 min), 8600 items vs Laya recipe (hours, 2×T4, ~30k
questions). Warm-start from Laya's encoder (`missing 0, unexpected 0`) bought
the jump; further gains need longer runs.

## 6. Unmeasured frontier

JevBench-fast here is 80 items (48 easy + 12 original + 20 hard) with n=1–4
families — direction, not a rank. Official composite (chance-corrected
Intelligence, Calibration, Speed, Cost) and the sealed hard tier are unrun.

Next, in order: (a) real task data + stronger teacher labels, (b) restore
`/tmp/tdb` and run full `bench_tdb.py` with a `frontier_v7` tag, (c) 1024-ctx
training, (d) RLCD/sequence objective, (e) full 231-item JevBench + sealed eval.
