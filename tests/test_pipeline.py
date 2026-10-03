"""E0 pipeline dry-run (CPU): materialize -> collate -> loss -> save/load.

Catches runtime contract breaks (shapes, keys, dtypes) without a GPU.
Run: python tests/test_pipeline.py
"""
import json
import sys

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
import numpy as np
import random
import torch


def test_materialize_collate_loss():
    from transformers import AutoTokenizer
    from train_v8 import materialize
    from common import collate, QTYPES
    from train_v8 import loss_v8

    tok = AutoTokenizer.from_pretrained(
        "/teamspace/studios/this_studio/frontier/frontier_ckpt_v7/tokenizer")
    rng = random.Random(0)
    recs = [
        {"state": f"Invoice {i} billed twice, refund requested.",
         "q": {"t": "choice", "ins": "Which?",
               "crit": {"billing": "charges", "technical": "bugs"}},
         "gold": i % 2, "teacher": None, "alpha": 0.0,
         "group": f"g{i % 3}", "source": "t"}
        for i in range(9)
    ] + [
        {"state": "Server down, all users blocked.",
         "q": {"t": "score", "ins": "Urgent?",
               "crit": ["low", "mid", "high"]},
         "gold": 2, "teacher": [0.5, -0.5, 1.2], "alpha": 0.5,
         "group": "g0", "source": "t"},
        {"state": "Please refund me.",
         "q": {"t": "noul", "ins": "Refund?", "crit": {}},
         "gold": 1, "teacher": None, "alpha": 0.0,
         "group": "g1", "source": "t"},
        {"state": "x",
         "q": {"t": "choice", "ins": "Which?",
               "crit": {f"opt{j}": None for j in range(5)}},
         "gold": 4, "teacher": [0.1, 0.2, 0.3, 0.4, 2.0], "alpha": 0.2,
         "group": "g2", "source": "t"},
    ]
    items = materialize(recs, tok, 512, 192, rng)
    assert len(items) == len(recs), f"dropped {len(recs) - len(items)}"
    # variable-k batch (the v6 crash): k=2,3,5,2 mixed
    sel = [e for e, _, _, _ in items]
    b = collate([[it] for it in sel], tok.pad_token_id)
    B = len(sel)
    assert b["input_ids"].shape[0] == B
    assert b["marker_mask"].shape[0] == B
    torch.manual_seed(0)
    K = b["marker_mask"].shape[1]
    logits = torch.randn(B, K)
    gold = b["label"]
    # pad teacher rows to kmax like the trainer
    rows = []
    for (e, tl, al, r) in items:
        k = len(e["markers"])
        v = torch.from_numpy(tl[:k]).float() if tl is not None else torch.full((k,), -1e4)
        if len(v) < K:
            v = torch.nn.functional.pad(v, (0, K - len(v)), value=-1e4)
        rows.append(v)
    tlog = torch.stack(rows)
    alpha = torch.tensor([a for _, _, a, _ in items])
    loss, parts = loss_v8(logits, gold, tlog, alpha, b["marker_mask"], b["qtype"])
    assert torch.isfinite(loss).all(), parts
    assert set(parts) == {"ce", "brier", "kl"}, parts
    print("PIPELINE_DRYRUN_OK", {"loss": round(loss.item(), 4), **parts})


def test_ckpt_roundtrip():
    from safetensors.torch import save_file, load_file
    import tempfile, os
    sd = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    with tempfile.TemporaryDirectory() as d:
        save_file(sd, f"{d}/m.safetensors")
        back = load_file(f"{d}/m.safetensors")
        assert torch.equal(back["w"], sd["w"])
    print("CKPT_ROUNDTRIP_OK")


if __name__ == "__main__":
    test_materialize_collate_loss()
    test_ckpt_roundtrip()
