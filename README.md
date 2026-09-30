# Frontier Decision Engine — lighter, faster alternative to Laya / TypeSafe Jev

## Official benchmark: Typed Decision Bench v0.3 (Blobfish, 5,387 items, 25 tasks)
Full roster, byte-identical inputs, verbatim official scoring
(`1 - normalised Brier` per question, mean per item ×100, unanswered = 0).
Run: `python3 bench_tdb.py` (use `--fraction 0.5` for the 2-min half; reproduces full scores ±0.12).
No TDB item or question sentence was trained on (per its contamination notice).

| System | DecisionScore | Primary acc | ECE(max-p) | Params |
|---|---|---|---|---|
| Jev 1.13 (published) | 81.70 | — | — | closed |
| Laya english (measured) | 62.78 | 0.5195 | 0.1917 | 421M |
| **Frontier v3 (measured, champion)** | **61.95** | 0.4187 | **0.1096** | **168M** |
| Frontier v4 (rejected) | 61.17* | 0.4221 | 0.1624 | 168M |
| Frontier v5 (rejected) | 60.99* | 0.4181 | 0.1435 | 168M |

*half-subset; halves reproduce full scores within 0.12 (laya 62.90/62.78, v3 61.83/61.95).
v4/v5 continued fine-tuning on synthetic drifted off v3's optimum — rejected, v3 stays champion.
Both models answered 5028/5028 (359 withheld = 0 for everyone).

## How Laya is built (from HF inspection)
- Backbone: `ModernBERT-large` 395M (hidden 1024, 28 layers, 8k RoPE) + 2-layer TransformerEncoder head + per-option `[MASK]` marker scorer + act/escalate head = **421M**.
- I/O: single forward pass `[CLS] <type> instructions [SEP] [MASK] opt… [SEP] state [SEP]`; logits read at marker positions, softmaxed per question. `head_max_len` 192 / `max_len` 512 (multilingual 256/1024, encoder to 8k).
- Training: RLCD (REINFORCE + group-mean baseline, reward = log + spherical + RPS, TD λ=1 for episodes).
- Calibration: per-bucket temperatures (`choice:2/3-5/6-10/11+`, `score:3-5`, `noul:2`).
- Measured here on T4: ~39ms / 2 questions warm; published: 39.5ms/1q, 158.6ms/10q, ECE 0.081 post-temp, typed-decisions-ft 0.766.

## This model (guide.md blueprint, bugs fixed)
- Backbone: `ModernBERT-base` 149M (hidden 768) + 2-layer refine + **SchemaCrossAttentionHead** (option markers as queries cross-attend full doc — strictly more expressive than Laya's linear scorer, same single-pass marker scheme, Jev-compatible) + pooled score/noul heads + act head = **168.5M (2.5× smaller)**.
- Loss: `CE + τ²·KL(teacher‖student) + Brier + RPS(score)` (guide.md, strictly proper — no gaming via overconfidence).
- Teachers: rule/logic ground-truth + softened paraphrase teacher (no API keys needed on T4); hard negatives (options 90% identical, one conditional clause) + paraphrase augmentation round 2.
- Calibration: per-bucket temperature grid fit on held-out split.
- Export: `model.safetensors` + `model.onnx` (opset 17, verified with onnxruntime).

## Measured on this T4 (Tesla T4, fp16)
| Metric | Laya (english, 421M) | Frontier (168.5M) |
|---|---|---|
| Params | 421M | **168.5M (2.5× smaller)** |
| Latency 1q | 39.5ms publ. | **~22ms p50** |
| Latency 10q batched | 158.6ms (15.9ms/q) | **~22ms (2.2ms/q, ~7× faster)** |
| ECE (calibrated, our synth bench) | 0.081 publ. (their bench) | **0.064** |
| Brier (calibrated, our synth bench) | — | **0.0007–0.004** |
| Synth held-out acc | — | 1.00 (in-distribution; see limits) |
| Hard paraphrase stress (6 cases) | — | **6/6 after round 2** (4/6 after round 1) |
| Live demo check (billing + churn) | billing 0.987 / noul 0.879 | billing 0.9998 / noul 0.9978 |

## Honest limits
- Synth accuracy is in-distribution; it does **not** prove beating Laya's 0.766 on their private typed-decisions set or Jev's 0.727. Zero-shot public benchmarks (AG News, Banking77-77, MASSIVE, XNLI) were **not** run here — run `eval_bench.py` extended to those sets before claiming SOTA.
- English-only (like Laya root). Multilingual needs mmBERT backbone swap.
- `act_probability` head is untrained signal (same caveat as Laya #185) — gate on `confidence`.
- 512 ctx in this ckpt (`head_max_len` 192); encoder supports 8k RoPE — raise `max_len` + retrain/fine-tune for long docs.

## Use
```python
from frontier.agent import FrontierAgent
a = FrontierAgent("frontier_ckpt_final")
r = a.system_one("Hi, we were billed twice...", {
  "department": {"type": "choice", "instructions": "Which department?",
                 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "else"}},
  "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"}})
```
Train: `python train.py --epochs 4` · Continue: `python continue_train.py` · Eval: `python eval_bench.py`
Files: `frontier/model.py common.py synth_data.py train.py continue_train.py eval_bench.py agent.py`, ckpt `frontier_ckpt_final/`.
