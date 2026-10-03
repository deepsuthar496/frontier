# Frontier Bottleneck Fix Plan

Repository: `deepsuthar496/frontier`  
Inspected snapshot: `7e29efbcdd8c9babfc2abf5c53de0b764819959b`  
Purpose: improve hard-task generalization while preserving a compact, non-autoregressive typed-decision model.

All proposed hyperparameters and acceptance thresholds below are experimental starting points, not measured results.

## 1. Architecture and current diagnosis

The inspected code exposes this implementation structure:

| Component | Relevant files | Observed role |
|---|---|---|
| Encoder and decision head | `model.py` | `FrontierDecisionEngine` and `SchemaCrossAttentionHead`; batched option queries cross-attend document representations and produce option scores. |
| Auxiliary action output | `model.py` | An `act_head` consumes pooled representations and additional features. Its complete training and inference behavior still needs inspection. |
| Input construction | `common.py` | State serialization, typed-option rendering, sequence construction, optional option ordering, and truncation controls. |
| Inference adapter | `agent.py` | `FrontierAgent` translates external question definitions into the internal representation and imports temperature-bucket utilities. |
| Gold and hard-task data | `data_v8.py`, `data_v9.py` | Public classification data, synthetic records, deterministic probability probes, grouped splitting, and JevBench-public ingestion. |
| Teacher preparation | `label_v9.py` | Offline Laya teacher-logit cache, padded to `KMAX=8`, aligned to training order. |
| Current training | `train_v9_ddp.py` | Gold-supervised loss plus teacher KD; four partitions; a separate materialization path. |
| Evaluation and diagnostics | `bench_full.py`, `bench_jevbench.py`, `audit_v8.py`, `eval_public.py` | Typed answer extraction, proper scoring, public benchmark evaluation, and truncation auditing. |
| Evidence-selection experiments | `evidence.py`, `test_evidence.py` | Chunking and chunk-scoring inference paths. |

This is an encoder-based decision system, not an autoregressive chatbot. A fix that requires generating long chain-of-thought text at inference would change the product’s compute and latency characteristics.

ModernBERT is designed for classification and retrieval, with native support for up to 8,192 tokens. That supports testing longer supervised inputs without replacing the backbone, but it does not establish that full fine-tuning at that length fits your T4 budget. [huggingface](https://huggingface.co/docs/transformers/model_doc/modernbert)

Three claims in the existing bottleneck record need tightening:

- Teacher ceiling: matching a teacher can constrain learning, but “distillation cannot surpass a wrong teacher” is too absolute. Research demonstrates smaller students surpassing teachers on particular tasks when given additional supervision. This is evidence for better supervision—not a guarantee for Frontier. [arxiv](https://arxiv.org/abs/2305.02301)
- Calibration attribution: a proper-score gap is not proof that calibration explains most of the difference. Separate discrimination, confidence, and scoring-implementation effects before assigning causality.
- Long-context conclusion: flat 512→1024 results do not establish that context is irrelevant when audited examples contain approximately 2,200 state tokens, before accounting for the question and options.

## 2. Correctness gates before training

These checks take priority over more training data or another checkpoint.

### Answer mapping audit

The retrieved `audit_v8.jsonl` excerpts contain records such as:

```text
id: hard-sol-a-multi_hop-10

labels:
  [deny_sensitive, needs_field_review,
   reduce_to_30_days, approve_45_days]

expected: approve_45_days
gold_idx: 0
```

Another excerpt has:

```text
id: hard-sol-b-routing_hard-01

labels:
  [spreadsheet_formula, office_file_repair,
   data_pipeline, desktop_support, finance_review]

expected: office_file_repair
gold_idx: 3
```

These are inconsistent if `gold_idx` indexes the displayed `labels`. They may be legitimate if the index refers to a separately permuted internal order, but the audit record does not make that mapping explicit.

Do not conclude that the reported standings are wrong yet. Determine which coordinate system each field uses.

Required changes:

- Log `canonical_options`, `model_options`, and `option_order`.
- Log both `gold_canonical_idx` and `gold_model_idx`.
- Decode predictions back to canonical answer values before comparing them with expected answers.
- Use the same mapping implementation for training, inference, audit, and benchmark scoring.
- Add round-trip tests covering every option permutation for small option sets.
- Handle `score`, `bool`, and `noul` according to their actual type semantics rather than assuming every output is an ordinary categorical choice.

Acceptance condition:

```python
assert canonical_options[gold_canonical_idx] == expected_answer
assert model_options[gold_model_idx] == expected_answer
assert decode_prediction(gold_model_idx, option_order) == expected_answer
```

For non-choice types, replace these with equivalent typed-value assertions.

### Teacher-cache integrity

`label_v9.py` describes a cache aligned to `data_v9.npz` training order. The retrieved training excerpt attaches teacher rows by index:

```python
it["teacher"] = [float(x) for x in _TL[i, :k]]
it["alpha"] = float(_AL[i])
```

Positional alignment needs a verifiable contract. Regeneration, sorting, filtering, or option-order changes must not silently attach the wrong distribution to an example.

Required changes:

- Give every example a stable content ID.
- Hash the state, question, type, criteria, canonical options, and label representation.
- Store IDs and option-order metadata beside teacher logits.
- Validate IDs, option counts, and option identities before merging.
- Fail on a mismatch; do not continue with a warning.
- Store dataset hash, teacher revision, serialization version, and teacher context length.

Acceptance condition:

```python
assert example.id == teacher_row.id
assert example.option_hash == teacher_row.option_hash
assert example.num_options == teacher_row.num_options
```

### Resolve configuration drift

The retrieved `train_v9_ddp.py` header says:

```text
alpha = 0.2 iff teacher argmax == gold,
otherwise 0.
```

But `label_v9.py` specifies:

```text
agree 0.5 / disagree 0.1
```

The training merge excerpt also reports `agree0.5` and `disagree0.1`.

That establishes documentation/configuration drift. It does not, from the excerpts alone, prove a loss-function bug.

Required changes:

- Define KD policy in one configuration object.
- Remove duplicated coefficients from comments and separate scripts.
- Persist the effective policy in the checkpoint.
- Log actual coefficient counts and measured CE/KL contributions.
- Add a test proving disagreement rows receive the configured coefficient.

### Protect evaluation independence

The retrieved `data_v9.py` excerpts confirm JevBench-public gold ingestion and public-family shards reserved for `final`.

Those withheld public examples can be a useful internal holdout, but they are not sealed evidence once the benchmark has already informed development.

Required changes:

- Maintain separate `public_train`, `development`, `calibration`, and `locked_test` manifests.
- Split synthetic data by generator/template structure before producing instances.
- Keep paraphrases, counterfactual siblings, and shared source documents in one partition.
- Check exact and near-duplicate leakage.
- Document whether the checkpoint used to initialize a run previously saw any held-out examples.
- Label every result as public-trained, development, internal holdout, or externally sealed.

## 3. Fix the capability bottlenecks

| Bottleneck | Proposed intervention | What must demonstrate success |
|---|---|---|
| Hard reasoning | Solver-verified curricula and counterfactual examples | Better unseen-template hard-family accuracy. |
| Teacher dependence | Gold-first learning with an explicit disagreement ablation | Gains where Laya is wrong, without regressions where it is correct. |
| Long context | Evidence-complete supervised inputs, then an all-chunk architecture if necessary | Correct use of scattered evidence at realistic lengths. |
| Calibration | Global baseline, then regularized grouped temperatures | Better held-out proper scores without relying on test-set fitting. |
| Evaluation | Canonical mappings, locked manifests, paired uncertainty analysis | Reproducible improvements on independent examples. |
| Compute limits | Token-aware batches, profiling, cached supervision | More validated experiments per GPU-hour. |

### Hard reasoning data

Do not respond to v8’s hard-task regression by adding more generic AG News or emotion examples. Add task-specific supervision that exposes the operations currently missing.

Build four initial hard-data generators:

| Family | Generate | Verify |
|---|---|---|
| `multi_hop` | Entity chains, policy dependencies, cross-section conditions, distractors | A graph or rule solver returns the answer and required evidence. |
| `probability` | Conditional probability, base rates, independence/dependence, expected value | Exact calculations and explicit rounding conventions. |
| `routing_hard` | Conflicting cues, precedence rules, exceptions, missing prerequisites | A deterministic policy evaluator returns the selected route. |
| `judge_hard` | Rubrics with necessary conditions, disqualifiers, partial fulfillment | A rubric evaluator returns the typed judgment. |

Every generated example should carry:

```text
example_id
template_id
split_group
family
difficulty
state
question
options
gold_answer
solver_version
supporting_evidence_ids
intermediate_targets
```

Start with a proposed 6,000–10,000 verified hard examples distributed across these families. Measure coverage and quality before increasing volume.

Counterfactual pairs are especially important:

```text
Rule:
Authenticated recurrence within 30 days is covered.

Case A:
Recurrence on day 20, authentication complete.
Answer: covered_recurrence

Case B:
Recurrence on day 20, authentication missing.
Answer: needs_authentication
```

Keep both cases in the same partition. Change distractor wording and answer position independently of the decisive condition.

### Learn intermediate structure

Research on step-by-step distillation supports using rationales as additional training supervision. However, the cited work trains rationale-generating models; it does not establish that Frontier’s encoder will benefit from copying that procedure unchanged. [arxiv](https://arxiv.org/abs/2305.02301)

For Frontier, test structured auxiliary supervision instead:

- Evidence identification.
- Applicable-rule prediction.
- Prerequisite satisfaction.
- Exception detection.
- Intermediate numerical quantities.
- Final answer selection.

Illustrative objective:

\[
L =
L_{\text{answer}}
+\alpha_i\tau^2D_{\mathrm{KL}}(p_T^\tau\parallel p_S^\tau)
+\lambda_eL_{\text{evidence}}
+\lambda_rL_{\text{rule}}
\]

Add one auxiliary objective at a time. Use training-only heads when they can be removed without changing final inference.

Never insert gold-derived rationales or solver outputs into evaluation inputs unless the deployed system can produce those inputs independently.

### Teacher disagreement policy

Run these ablations from the same checkpoint and dataset:

| Policy | Teacher agrees with verified gold | Teacher disagrees |
|---|---:|---:|
| Gold-only | 0 | 0 |
| Conservative KD | 0.2 | 0 |
| Current-policy comparison | 0.5 | 0.1 |

Treat these as starting values, not optimal coefficients.

On solver-verified rows, gold-only disagreement handling should be the first comparison. A smaller nonzero KD weight can still pull the student toward a wrong teacher.

Report four slices:

- Teacher correct, student correct.
- Teacher correct, student wrong.
- Teacher wrong, student correct.
- Both wrong.

The third slice directly tests whether Frontier learns useful information beyond Laya.

When generating teacher labels, log whether the teacher saw all required evidence. Teacher confidence on a truncated input is not evidence that its target distribution is reliable.

### Long-context training

The audit excerpts include multi-hop states around 2,200 tokens. A 1,024-token total sequence cannot preserve all such states plus instructions and options.

Required changes:

- Measure full serialized sequence length—not state length alone.
- Track retained supporting evidence, not just retained token fraction.
- Prevent silent truncation of schema and option text.
- Train on examples with evidence distributed across beginning, middle, and end.
- Include examples where multiple distant facts are jointly necessary.
- Test 2,048 and 3,072 total tokens if profiling permits.

ModernBERT’s supported context length makes these architectural tests plausible; actual memory and throughput must be measured on your environment. [huggingface](https://huggingface.co/docs/transformers/model_doc/modernbert)

If adequate full-sequence training cannot fit the budget, test a hierarchical model that encodes every chunk and learns cross-chunk aggregation. Preserve chunk positions and evidence relationships.

That is different from the failed evidence-ranking path: it retains representations from all chunks rather than dropping “low-ranked” facts.

### Calibration

Do not make “thousands per bucket” a universal rule. Calibration requirements depend on data quality, distribution, and model flexibility. Research shows temperature scaling can be unreliable with small or noisy validation sets. [arxiv](https://arxiv.org/html/1810.11586)

Implement:

1. Global temperature as the baseline.
2. Grouped temperatures by supported question type and option count.
3. Regularization toward the global temperature.
4. Held-out comparison before enabling a grouped parameter.

Suggested fitting objective:

\[
\min_{\{T_b\},T_g}
\sum_i-\log p_{T_{b(i)}}(y_i\mid x_i)
+\lambda\sum_b(\log T_b-\log T_g)^2
\]

This is a proposed regularized design, not a published Frontier result.

Report NLL, Brier score, the harness proper score, ECE, and reliability plots. Specify the Brier normalization convention and ordinal-score handling.

A positive scalar temperature preserves argmax. Therefore, temperature scaling cannot fix hard-task accuracy; it can only alter the probability distribution.

## 4. Implementation and experiment order

### Proposed developer changes

| File or module | Change |
|---|---|
| `common.py` | Centralize canonical option mapping, typed decoding, sequence budgeting, and serialization versioning. |
| `audit_v8.py` | Record canonical/internal mappings and supporting-evidence coverage. |
| `data_v8.py`, `data_v9.py` | Add stable IDs, structural split groups, solver metadata, and contamination checks. |
| `label_v9.py` | Store content and option hashes; merge by validated ID rather than trusting position alone. |
| `train_v9_ddp.py` | Centralize KD policy, log component losses, and make hard-family sampling explicit. |
| `model.py` | Add optional training-only evidence/rule heads behind configuration flags. |
| `agent.py` | Apply the shared decoder and validated calibration configuration. |
| `bench_full.py` | Save per-item canonical predictions and paired comparison inputs. |
| `evidence.py` | Keep experimental selection separate from the default benchmark path. |
| New tests | Mapping, masking, teacher alignment, split leakage, scoring, and distributed-loss equivalence. |

These are proposed edit targets. Full file bodies must be inspected before making changes.

### Compute discipline

Use single-T4-compatible defaults, with two-GPU DDP as an optional execution mode.

- Generate solver labels on CPU.
- Cache teacher outputs offline.
- Cache tokenization with dataset and tokenizer hashes.
- Bucket batches by token length and option count.
- Use mixed precision supported by the actual environment and gradient checkpointing where beneficial.
- Profile memory and throughput before long-context runs.
- Compare equal optimizer-update counts and effective global batches.
- Save optimizer, scheduler, scaler, RNG, and sampler state for resumable runs.
- Record peak memory, examples/second, tokens/second, and GPU-hours.

Do not accept “DDP exact math” solely because a run finished faster. Test distributed gradients against a single-process reference, including unequal final batches and accumulation boundaries.

### Minimal experiment ladder

| Run | Change | Purpose |
|---|---|---|
| E0 | Mapping, scoring, cache, and leakage tests | Establish trustworthy infrastructure. |
| E1 | Reproduce the selected baseline | Confirm that the current code reproduces saved results. |
| E2 | Verified hard data; gold-only learning | Isolate supervision improvements. |
| E3 | Add conservative agreement-only KD | Measure the value of teacher distributions. |
| E4 | Compare current 0.5/0.1 policy | Measure whether disagreement KD helps or hurts. |
| E5 | Longer evidence-complete inputs | Isolate context coverage. |
| E6 | Add one structured auxiliary objective | Test intermediate supervision. |
| E7 | Calibrate the chosen checkpoint | Improve probabilities without changing task learning. |

Do not run every combination immediately. Advance only interventions that help the development hard set while preserving emotion performance.

For promising interventions, repeat with multiple seeds and report variation. Do not select the best seed using the locked test.

## 5. Release acceptance and reporting

Use the pasted standings as historical project measurements, not newly reproduced results.

The release report should distinguish:

| Evaluation | Permitted interpretation |
|---|---|
| Full-231 after public-gold training | Public-trained diagnostic performance. |
| Withheld shards of already-used public benchmark | Internal holdout, not sealed evaluation. |
| Locked unseen templates and source groups | Evidence of generalization within the constructed distribution. |
| Untouched external sealed benchmark | Stronger external generalization evidence. |
| Historical TDB scores | Historical only until roster/version and evaluation are reproduced. |
| Published Jev score | Published claim, not a directly measured Frontier comparison. |

Proposed acceptance gates:

- All answer-mapping, teacher-alignment, masking, and split-integrity tests pass.
- Emotion accuracy remains at least 0.80 on the agreed fixed evaluation set.
- Hard-family gains appear on unseen structural templates, not only familiar wording.
- The hard-family aggregate improves over Laya on the same paired independent examples.
- Confidence intervals and item counts accompany family scores.
- Proper-score gains are reported separately from accuracy gains.
- Latency is measured on the same hardware, precision, batch size, and input-length distribution.
- Any tool-assisted numerical or rule-solving variant is reported separately from model-only Frontier.

For accuracy, use paired item-level comparisons and an appropriate paired test. For proper scores, bootstrap paired score differences; when examples share templates or documents, resample those groups rather than pretending all rows are independent.

The developer’s first deliverable should be an audit answering:

```text
1. Are logged labels and gold indices in the same coordinate system?
2. Can every prediction round-trip to the expected canonical value?
3. Does every teacher row match the exact example and option order?
4. Which KD coefficients does the executed loss actually use?
5. Which required evidence survives serialization and truncation?
6. Which evaluation items or structural templates influenced training?
7. Can the saved baseline be reproduced from the pinned snapshot?
```

Only after those answers are trustworthy should v9’s learning changes be judged. The strongest candidate path is verified hard-task supervision plus complete evidence exposure, with conservative KD and structured auxiliary targets tested as separate ablations—not another blanket distillation run.