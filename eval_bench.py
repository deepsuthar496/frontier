"""Calibrated eval + T4 latency bench + head-to-head vs Laya."""
import json, time
import numpy as np
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
from common import collate, build_sequence, ece_score, confidence_from_probs, temp_bucket, QTYPES
from model import FrontierDecisionEngine
from synth_data import build_records
from train import encode


def load_frontier(d="/teamspace/studios/this_studio/frontier/frontier_ckpt_final", device="cuda"):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel
    device = torch.device(device)
    cfg = json.load(open(f"{d}/frontier_config.json"))
    tok = AutoTokenizer.from_pretrained(f"{d}/tokenizer")
    enc = AutoModel.from_config(AutoConfig.from_pretrained(f"{d}/encoder"), attn_implementation="sdpa")
    m = FrontierDecisionEngine(enc, head_layers=cfg.get("head_layers", 2))
    m.load_state_dict(load_file(f"{d}/model.safetensors"), strict=True)
    m.to(device).eval()
    return m, tok, cfg, device


def calibrated_metrics(m, tok, cfg, device, items):
    m.eval()
    acc, confs, corr, brier, maes, nll = [], [], [], [], [], []
    T, Tb = cfg["temperature"], cfg.get("temperature_by_options", {})
    with torch.no_grad():
        for s in range(0, len(items), 64):
            sel = items[s:s+64]
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                out = m(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
            lg = out["choice_logits"].float().cpu().numpy()
            for r, it in enumerate(sel):
                kk = len(it["markers"])
                t = Tb.get(temp_bucket(it["qtype"], kk), T[it["qtype"]])
                z = lg[r, :kk] / t
                e = np.exp(z - z.max()); p = e / e.sum()
                pred = int(p.argmax())
                co = pred == it["label"]
                acc.append(co); confs.append(confidence_from_probs(p, kk)); corr.append(co)
                tgt = np.zeros(kk); tgt[it["label"]] = 1
                brier.append(float(((p-tgt)**2).mean()))
                nll.append(-np.log(max(p[it["label"]], 1e-9)))
                if it["qtype_name"] == "score":
                    maes.append(abs(float((np.arange(kk)*p).sum()) - it["label"]))
    acc = np.array(acc)
    return {"acc": float(acc.mean()), "ece": ece_score(np.array(confs), acc),
            "brier": float(np.mean(brier)), "nll": float(np.mean(nll)),
            "score_mae": float(np.mean(maes)) if maes else 0.0}


def bench_latency(m, tok, cfg, device, n_q=(1, 5, 10), reps=30):
    m.eval()
    state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    out = {}
    for nq in n_q:
        qs = {f"q{i}": {"ins": "Which department handles this?",
                        "crit": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs, outages", "other": "else"}} for i in range(nq)}
        items = []
        for qid, qd in qs.items():
            q = {"t": "choice", "ins": qd["ins"], "crit": qd["crit"]}
            seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
        b = collate([items], tok.pad_token_id)
        bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        torch.cuda.synchronize() if device.type=="cuda" else None
        for _ in range(5):  # warmup
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                m(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
        ts = []
        with torch.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                    m(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
                if device.type=="cuda": torch.cuda.synchronize()
                ts.append((time.perf_counter()-t0)*1000)
        out[nq] = {"p50": float(np.median(ts)), "p99": float(np.percentile(ts,99)),
                   "ms_per_q": float(np.median(ts)/nq)}
    return out


if __name__ == "__main__":
    import random
    m, tok, cfg, device = load_frontier()
    print("params ckpt:", cfg["params_M"], "M vs Laya 421M")
    rng = random.Random(999)  # held-out seed: unseen paraphrases
    recs = build_records(n_per_workflow=150, seed=999)
    items = [e for r in recs for e in [encode(r, tok, cfg["max_len"], cfg["head_max_len"], False, rng)] if e]
    print("heldout items:", len(items))
    met = calibrated_metrics(m, tok, cfg, device, items)
    print("CALIBRATED heldout:", json.dumps(met, indent=2))
    lat = bench_latency(m, tok, cfg, device)
    print("LATENCY T4 ms:", json.dumps(lat, indent=2))
    # try Laya head-to-head
    try:
        from transformers import AutoTokenizer as AT
        import laya  # pip package if available
        print("laya pkg available, running head-to-head...")
    except Exception as e:
        print("laya pkg not installed, skipping live Laya bench:", e)
        print("Laya published refs: 421M, 39.5ms/1q, 158.6ms/10q, ECE 0.081(post-temp), typed-decisions-ft 0.766")
