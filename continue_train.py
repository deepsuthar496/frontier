"""Warm-start continuation: load frontier_ckpt, train 3 more epochs on augmented data."""
import argparse, json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
from model import FrontierDecisionEngine, count_params
from common import collate, total_loss
from synth_data import build_records
from train import encode, make_batches, evaluate, fit_temperatures
import random, time

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="frontier_ckpt")
ap.add_argument("--out", default="frontier_ckpt_v2")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=1.5e-5)
a = ap.parse_args()

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(f"{a.ckpt}/tokenizer")
enc = AutoModel.from_config(AutoConfig.from_pretrained(f"{a.ckpt}/encoder"), attn_implementation="sdpa")
model = FrontierDecisionEngine(enc, head_layers=2)
from safetensors.torch import load_file
model.load_state_dict(load_file(f"{a.ckpt}/model.safetensors"), strict=True)
model.to(device)
print("warm start params:", round(count_params(model)/1e6,1), "M")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
recs = build_records(n_per_workflow=800, seed=7)
rng = random.Random(7)
cfg0 = json.load(open(f"{a.ckpt}/frontier_config.json"))
items = [e for r in recs for e in [encode(r, tok, cfg0["max_len"], cfg0["head_max_len"], True, rng)] if e]
cut = int(len(items)*0.8)
tr, va = items[:cut], items[cut:cut+600]
print(len(tr), len(va))
scaler = torch.amp.GradScaler("cuda")
for ep in range(a.epochs):
    model.train()
    batches = make_batches(tr, seed=100+ep)
    tot = 0; t0=time.time()
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
        scaler.step(opt); scaler.update(); opt.zero_grad()
    tot = loss.item()
    m = evaluate(model, va, tok, device, cfg0["max_len"], cfg0["head_max_len"])
    print(f"ep {ep+1} val acc {m['acc']:.3f} ece {m['ece']:.3f} brier {m['brier']:.4f} ({time.time()-t0:.0f}s)")

temps, temps_by = fit_temperatures(model, va, tok, device)
print(temps, temps_by)
import os
os.makedirs(f"{a.out}/encoder", exist_ok=True)
os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
from safetensors.torch import save_file
save_file(model.state_dict(), f"{a.out}/model.safetensors")
model.encoder.config.save_pretrained(f"{a.out}/encoder")
tok.save_pretrained(f"{a.out}/tokenizer")
cfg0.update({"temperature": temps, "temperature_by_options": temps_by})
json.dump(cfg0, open(f"{a.out}/frontier_config.json","w"), indent=2)
print("saved", a.out)
