# What Is Stopping Us From Frontier — Ranked Blockers

Measured context: Typed Decision Bench v0.3, full 5,387 roster —
Laya english 62.78 vs Frontier v3 61.95 (gap 0.83), primary accuracy 0.52 vs 0.42,
ECE 0.19 vs 0.11. Jev 1.13 published at 81.70 (not measured here).

## 1. No teacher (biggest blocker)
Laya trains with RLCD on **teacher gold distributions** (soft labels from frontier
models); JevFish distills too. We train on hard labels plus hand-made softening.
A student almost never beats its teacher's signal quality — our ceiling is our own
synthetic labels. Evidence: v4 (61.17) and v5 (60.99) proved more self-training
equals drift, not gains.
**Unlock:** use **Laya itself (Apache-2.0, local, no API needed) as a teacher** —
generate soft distributions from `laya` and `laya-typed-decisions` over our training
states, distill with KL. Fully doable on a T4; costs only inference time.

## 2. Our data is not the real distribution
v3 wins legal/finance tasks (+17 on trial-outcomes) but loses safety, tool-routing
and transcripts (−12 to −15). Synthetic injection tickets are not real
prompt-injection; faux transcripts are not agent trajectories. TDB's contamination
rule forbids training on its items or question sentences.
**Unlock:** teacher-label **real public data** — the 21 TDB source datasets are
openly licensed (CFPB complaints, HotpotQA, FiNER, etc.) — instead of inventing
lookalikes.

## 3. Argmax gap: 0.42 vs 0.52
Our probabilities are better shaped (ECE 0.11 vs 0.19) but our top pick is wrong
more often. At 168M with 2 head layers, 64-option and hierarchical questions
(dbpedia, finer) may need more capacity or a better head — the cross-attention idea
did not prove out where it matters.
**Unlock:** ablate head variants (scorer depth, option budget) **with the teacher
signal from §1**; consider a 200–300M backbone once data stops being the bottleneck.

## 4. Long context is untrained
The bench has 4,000-token long items plus 799 trajectories; we train at ≤512
(v4's 1024 attempt regressed). Both models truncate — Laya copes better, likely
from trajectory training (TD-λ) that we do not have.
**Unlock:** train at 1024–2048 on real long states with prefix/trajectory
objectives, not synthetic filler.

## 5. Calibration sample starvation
Proper bucket temperatures need 2,000+ samples per bucket (Laya's floor); our fits
run on hundreds and sit on clamp floors (0.5). ECE is good but fragile.
**Unlock:** calibrate on thousands of teacher-labeled items once §1 exists.

## 6. Budget reality: one T4 plus "don't make me wait"
Laya's recipe is 2×T4 for hours over ~30k questions. Our rounds are minutes on
crumbs. Frontier results need frontier-scale runs.

## 7. The actual frontier (Jev 81.70) is closed
We cannot measure it, only quote its published score. "Beating Jev" cannot be
verified from here regardless.

## Bottom line
Nothing exotic — it is **teacher signal + real task data + scale**. The single move
that unblocks the most: **distill Laya's open checkpoints as teachers over real
public task data** (their license permits it, our hardware can do it).
