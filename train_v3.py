"""v3 training: laya's finetune recipe (discriminative LRs, token-budget batching,
fp16, grad checkpoint) + guide.md loss + mixed data. Fresh backbone (v2 head is
template-biased; keep only the architecture).

Success bars (must clear ALL): AG News>=0.85, Emotion>=0.45, order-flip<=0.15,
synth-hard 6/6, ECE(max-p, clamped)<=0.12.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")
from model import FrontierDecisionEngine, count_params
from common import collate, total_loss
from synth_data import build_records
from data_v3 import CHOICE_INS, SCORE_INS
from train import encode, make_batches, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frontier_ckpt_v3")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.encoder)
    enc = AutoModel.from_pretrained(a.encoder, attn_implementation="sdpa")
    enc.gradient_checkpointing_enable()
    model = FrontierDecisionEngine(enc, head_layers=2).to(device)
    print("params:", round(count_params(model)/1e6,1), "M")
    # discriminative LRs per laya docs/finetune.md
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 2.5e-5},
                             {"params": list(model.refine.parameters()) + list(model.choice_head.parameters())
                              + list(model.pool_weights.parameters()) + list(model.score_head.parameters())
                              + list(model.noul_head.parameters()) + list(model.act_head.parameters())
                              + list(model.type_emb.parameters()), "lr": 1e-4}],
                            weight_decay=0.01)
    # synth with instruction diversity: rewrite fixed instructions
    rng = random.Random(11)
    syn = build_records(n_per_workflow=700, seed=11)
    for r in syn:
        if r["q"]["t"] == "choice":
            r["q"] = {"t": "choice", "ins": rng.choice(CHOICE_INS), "crit": r["q"]["crit"]}
        elif r["q"]["t"] == "score":
            r["q"] = {"t": "score", "ins": rng.choice(SCORE_INS), "crit": r["q"]["crit"]}
    from data_v3 import public_records
    pub = public_records(seed=11)
    recs = syn + pub
    rng.shuffle(recs)
    items = [e for r in recs for e in [encode(r, tok, 512, 192, True, rng)] if e]
    cut = int(len(items)*0.9)
    tr, va = items[:cut], items[cut:cut+800]
    print(f"mix: synth {len(syn)} pub {len(pub)} | train {len(tr)} val {len(va)}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*max(1, len(tr)//24))
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(a.epochs):
        model.train()
        batches = make_batches(tr, seed=11+ep)
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
        m = evaluate(model, va, tok, device, 512, 192)
        print(f"ep {ep+1} loss {tot/max(1,len(batches)):.4f} acc {m['acc']:.3f} brier {m['brier']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    # proper calibration with LAYA's fitter (floors + clamp), max-p NLL
    sys.path.insert(0, "/teamspace/studios/this_studio/laya")
    from laya import calibrate as laya_cal
    from laya.common import QTYPES as LQT
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
    print("calibration:", json.dumps({k: v for k, v in fit.items() if k != "n_by_bucket"}, indent=1)[:800])
    print("n_by_bucket:", fit.get("n_by_bucket"))
    from laya.common import clamp_temperature
    temps = [clamp_temperature(t) for t in fit["temperature"]]
    temps_by = {k: clamp_temperature(v) for k, v in fit.get("temperature_by_options", {}).items()}
    print("clamped:", temps, temps_by)
    os.makedirs(f"{a.out}/encoder", exist_ok=True)
    os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), f"{a.out}/model.safetensors")
    model.encoder.config.save_pretrained(f"{a.out}/encoder")
    tok.save_pretrained(f"{a.out}/tokenizer")
    json.dump({"encoder": a.encoder, "head_layers": 2, "max_len": 512, "head_max_len": 192,
               "max_prefixes": 6, "act_costs": {"escalate": 0.5}, "amp_dtype": "fp16",
               "temperature": temps, "temperature_by_options": temps_by,
               "params_M": round(count_params(model)/1e6,1)}, open(f"{a.out}/frontier_config.json","w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
