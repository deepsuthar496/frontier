"""v4: continue from v3 at max_len 1024 / head 256 (laya-typed-decisions geometry),
dict-state + safety + long + hierarchical data with replay. Low discriminative LRs."""
import argparse, json, os, random, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")
from model import FrontierDecisionEngine, count_params
from common import collate, total_loss
from train import encode, make_batches, evaluate
from data_v4 import build_v4
from synth_data import build_records
from data_v3 import public_records, CHOICE_INS, SCORE_INS

MAX_LEN, HEAD_LEN = 1024, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="frontier_ckpt_v3")
    ap.add_argument("--out", default="frontier_ckpt_v4")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(f"{a.src}/tokenizer")
    enc = AutoModel.from_config(AutoConfig.from_pretrained(f"{a.src}/encoder"), attn_implementation="sdpa")
    model = FrontierDecisionEngine(enc, head_layers=2)
    from safetensors.torch import load_file, save_file
    model.load_state_dict(load_file(f"{a.src}/model.safetensors"), strict=True)
    model.to(device)
    enc.gradient_checkpointing_enable()
    print("params:", round(count_params(model)/1e6,1), "M")
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-5},
                             {"params": list(model.refine.parameters()) + list(model.choice_head.parameters())
                              + list(model.pool_weights.parameters()) + list(model.score_head.parameters())
                              + list(model.noul_head.parameters()) + list(model.act_head.parameters())
                              + list(model.type_emb.parameters()), "lr": 5e-5}],
                            weight_decay=0.01)
    rng = random.Random(21)
    new = build_v4(seed=21)
    syn = build_records(n_per_workflow=250, seed=21)
    for r in syn:
        if r["q"]["t"] == "choice":
            r["q"] = {"t": "choice", "ins": rng.choice(CHOICE_INS), "crit": r["q"]["crit"]}
        elif r["q"]["t"] == "score":
            r["q"] = {"t": "score", "ins": rng.choice(SCORE_INS), "crit": r["q"]["crit"]}
    pub = public_records(n_ag=600, n_em=400, seed=21)
    recs = new + syn + pub
    rng.shuffle(recs)
    items = [e for r in recs for e in [encode(r, tok, MAX_LEN, HEAD_LEN, True, rng)] if e]
    cut = int(len(items)*0.9)
    tr, va = items[:cut], items[cut:cut+800]
    print(f"new {len(new)} replay {len(syn)+len(pub)} | train {len(tr)} val {len(va)}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*max(1, len(tr)//24))
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(a.epochs):
        model.train()
        batches = make_batches(tr, max_tokens=8192, max_seqs=16, seed=21+ep)
        t0 = time.time(); tot = 0
        for bidx in batches:
            sel = [tr[i] for i in bidx]
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            teacher = torch.log(bd["target"].clamp_min(1e-6)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
                loss, _ = total_loss(out["choice_logits"], teacher, bd["target"].to(device),
                                     bd["marker_mask"].to(device), bd["qtype"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            tot += loss.item()
        m = evaluate(model, va, tok, device, MAX_LEN, HEAD_LEN)
        print(f"ep {ep+1} loss {tot/max(1,len(batches)):.4f} acc {m['acc']:.3f} brier {m['brier']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    from laya import calibrate as laya_cal
    from laya.common import clamp_temperature
    model.eval()
    records = []
    with torch.no_grad():
        for s in range(0, len(va), 64):
            sel = va[s:s+64]
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
            lg = out["choice_logits"].float().cpu().numpy()
            for r, it in enumerate(sel):
                kk = len(it["markers"])
                tgt = np.zeros(kk); tgt[it["label"]] = 1
                records.append((int(it["qtype"]), lg[r, :kk].astype(float), tgt.astype(float), kk))
    fit = laya_cal.fit_temperature_map(records, compute_ece=True, seed=0)
    temps = [clamp_temperature(t) for t in fit["temperature"]]
    temps_by = {k: clamp_temperature(v) for k, v in fit.get("temperature_by_options", {}).items()}
    print("temps:", temps, temps_by, "n:", fit.get("n_by_bucket"))
    os.makedirs(f"{a.out}/encoder", exist_ok=True)
    os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
    save_file(model.state_dict(), f"{a.out}/model.safetensors")
    model.encoder.config.save_pretrained(f"{a.out}/encoder")
    tok.save_pretrained(f"{a.out}/tokenizer")
    json.dump({"encoder": "answerdotai/ModernBERT-base", "head_layers": 2, "max_len": MAX_LEN,
               "head_max_len": HEAD_LEN, "max_prefixes": 6, "act_costs": {"escalate": 0.5},
               "amp_dtype": "fp16", "temperature": temps, "temperature_by_options": temps_by,
               "params_M": round(count_params(model)/1e6,1)}, open(f"{a.out}/frontier_config.json","w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
