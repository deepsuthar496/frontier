# Kaggle Training Workflow (frontier project runbook)

How models get trained on Kaggle remote GPUs and moved back here.
Local box is CPU-only; Kaggle provides the GPUs (T4 x2 per kernel).

## 0. Auth (CLI 2.2.2+)

Classic `kaggle.json` no longer works (401). New flow needs the API token:

```bash
pip install "kaggle==2.2.2"
export KAGGLE_API_TOKEN=KGAT_<token>   # Settings -> API on kaggle.com
kaggle kernels list --mine             # verify
```

Token crossed chat once — rotate it in Settings when done. Never commit it.

## 1. What goes up: datasets (inputs)

Kernels ship ONLY `script.py`. Everything else rides as input datasets:

| Dataset | Contents | Size | Built by |
|---|---|---|---|
| `frontier-v8-src` | frontier/*.py + `laya/` pkg + `data_v8/v9.npz` | ~6MB | `kaggle datasets version --dir-mode zip -p <dir>` |
| `frontier-v8-ep1-fp16` | ep1 weights fp16 + config + tokenizer | ~860MB | `kaggle datasets create` (one-shot) |
| `frontier-bench-weights` | v7 + v8ddp fp32 ckpts | ~3.4GB | `kaggle datasets create` (one-shot) |
| `frontier-v9-teacher` | `teacher_v9.npz` (Laya logits) | ~200KB | label kernel output re-uploaded |
| `frontier-v9-ckpt-fp16` | v9 ep2 fp16 + config | ~860MB | from train outputs |

Rules learned the hard way:
- `--dir-mode zip` for folders (bare `create` skips subdirs silently).
- Version bumps (`datasets version`) must finish processing before a kernel
  references new files — poll `datasets files` for the new filename.
- Kernel file layout is NEVER as pushed: `find()` walk `/kaggle/input`
  for real paths instead of assuming them.
- `/tmp` on the local box is wiped on restarts — rebuild bundles from the repo.

## 2. Kernels (one at a time)

| Kernel | Job | In | Out |
|---|---|---|---|
| `frontier-v8-smoke` | validate CUDA count, HF pulls, imports | — | log only |
| `frontier-v8-train` | v8 512, 2ep, single GPU | src + ep1? (init_hf) | ckpt + fp16 |
| `frontier-v8-1024` | 1024 continue, single GPU | src + ep1-fp16 | ckpt |
| `frontier-v8-ddp` | 1024, torchrun 2 procs, NCCL | src + ep1-fp16 | ckpt + fp16 |
| `frontier-v9-label` | Laya logits over 27k (agree .5/.1) | src | `teacher_v9.npz` |
| `frontier-v9-train` | hybrid gold+KD, DDP 2ep 1024 | src + ep1 + teacher | ckpt + fp16 |
| `frontier-bench-full` / `-v9` | 231-item bench (+ensemble) | src + weights | `bench_*.json` |
| `frontier-final-eval` | locked-split eval | src + weights | `final_eval.json` |
| `frontier-kev-teacher` | Kev-4B logits over 27k | src | `teacher_kev.npz` |
| `frontier-evidence` | long-family selection eval | src + ep1 | `evidence_full.json` |

Push / poll / pull:

```bash
kaggle kernels push -p /tmp/kegXYZ            # new version + run
kaggle kernels status USER/SLUG               # RUNNING / COMPLETE / ERROR
kaggle kernels logs USER/SLUG                 # live-ish log (JSON stream)
kaggle kernels output USER/SLUG -p OUT --force # /kaggle/working files
```

There is NO remote kill. A bad kernel runs to completion — budget for it
(~30h GPU/week; typical run 30-60 min). Fix-forward with a new version.

## 3. Working DDP pattern (2xT4, one kernel)

- `torchrun --nproc_per_node=2 --standalone train_ddp.py` from a launcher
  script (subprocess; launcher env is NOT inherited — pass `PYTHONPATH` via
  `env=` explicitly).
- Block-strided microbatch split so both ranks hit every sync step together
  (naive striding NCCL-hangs).
- `find_unused_parameters=True` (aux heads unused by loss) — first attempt
  without it died in the reducer.
- Rank-0-only eval/save + barriers; NCCL timeout 30 min (rank-0 eval is long).
- Sharded eval (each rank scores half, `all_gather` combines) — a 10-min
  one-sided barrier wait kills the job at the default 10-min NCCL timeout.
- Measured: 26 min DDP vs 45+ single for the same epoch, exact same dev.

## 4. Failure modes seen (and fixes)

| Symptom | Cause | Fix |
|---|---|---|
| `No module named X` in workers | launcher `sys.path` not inherited | `PYTHONPATH` in subprocess env |
| DDP reducer error, params 209-222 | aux heads unused by loss | `find_unused_parameters=True` |
| NCCL watchdog 600s, SIGABRT | one-sided eval vs barrier | sharded eval + 30-min timeout |
| `Unrecognized model in encoder` | newer-transformers config | rewrite config from HF |
| Tokenizer backend error | same serialization skew | refresh tokenizer from HF |
| Pin `transformers==5.17.0 tokenizers==0.23.2` in every kernel | floating `-U` breaks loads | exact pins |
| Kev OOM at load | fp32 LoRA merge transient | unmerged bf16 (`LoadOptions`) |
| Kev `'label'` KeyError | record needs per-q `label`+`src` | Jev shape + label/src |
| Kev torchao pin | image ships 0.10, needs >0.16 | `pip install torchao>=0.16` |
| Score crit `.keys()` crash | score crit is a list | branch by `q["t"]` |

## 5. Local loop (CPU box)

- Download: `kaggle datasets download SLUG` or `kernels output` (fp16 for
  eval, fp32 only if continuing training).
- fp16 loads into the fp32 graph via `copy_` cast — eval-identical, half bytes.
- Evals run on CPU (slow, ~3s/item at 429M): JevBench-fast-80, sealed-60,
  AG/Emotion. GPU numbers come from bench kernels instead.
- Commit code + result JSONs; checkpoints stay local (gitignored).
- `/tmp` is ephemeral — kernels scripts live in repo or get rebuilt.

## 6. Quota discipline

- One kernel at a time (each holds its own 2xT4 box; overlaps burn 2x quota).
- Snapshot per epoch + fp16 export every run (a dead kernel's outputs are
  still downloadable — v4's weights were recovered this way).
- Label/cache once, train many times (teacher npz reused across runs).
