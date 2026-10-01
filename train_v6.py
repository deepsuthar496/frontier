"""v6: Teacher-1 distillation (spec section 4) with REAL Laya soft logits.

Loss (spec formula): CE(hard) + tau^2 * KL(teacher || student) + Brier + RPS(score).
From v3 (champion), 512/192 geometry, 90/10 pre-split, discriminative LRs,
laya-fitter calibration. Promote ONLY if TDB-half > 61.83 and JevBench improves.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")
from model import FrontierDecisionEngine, count_params
from common import collate, total_loss
from train import encode as encode_rec, make_batches, evaluate


def load_teacher(path="teacher_labels.npz"):
    z = np.load(path, allow_pickle=True)
    recs = json.loads(str(z["recs"]))
    return recs, z["teacher_logits"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="frontier_ckpt_v3")
    ap.add_argument("--out", default="frontier_ckpt_v6")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tau", type=float, default=2.0)
    a = ap.parse_args()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(f"{a.src}/tokenizer")
    enc = AutoModel.from_config(AutoConfig.from_pretrained(f"{a.src}/encoder"), attn_implementation="sdpa")
    model = FrontierDecisionEngine(enc, head_layers=2)
    from safetensors.torch import load_file, save_file
    model.load_state_dict(load_file(f"{a.src}/model.safetensors"), strict=True)
    model.to(device)
    enc.gradient_checkpointing_enable()
    print("params:", round(count_params(model)/1e6,1), "M", flush=True)

    recs, TL = load_teacher()
    print(f"teacher labels: {len(recs)} records", flush=True)
    rng = random.Random(41)
    # materialize student items WITHOUT reshuffling (order locked to teacher)
    from common import QTYPES
    items = []
    for i, r in enumerate(recs):
        q = r["q"]
        e = encode_rec({"state": r["state"], "q": {"t": q["t"], "ins": q["ins"], "crit": q["crit"]},
                        "soft": r["soft"], "y": r["y"]},
                       tok, 512, 192, False, rng)
        if e is None:
            continue
        items.append((e, TL[i]))
    # split AFTER materialize so train/val map to actual items, not recs indices
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(len(idx)*0.9)
    tr_idx, va_idx = set(idx[:cut]), set(idx[cut:])
    tr = [items[i] for i in idx[:cut]]
    va = [items[i] for i in idx[cut:]]
    print(f"train {len(tr)} val {len(va)} (dropped {len(recs)-len(items)})", flush=True)

    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-5},
                             {"params": list(model.refine.parameters()) + list(model.choice_head.parameters())
                              + list(model.pool_weights.parameters()) + list(model.score_head.parameters())
                              + list(model.noul_head.parameters()) + list(model.act_head.parameters())
                              + list(model.type_emb.parameters()), "lr": 5e-5}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*max(1, len(tr)//24))
    scaler = torch.amp.GradScaler("cuda")
    # batch by student item lengths
    lens = np.array([len(e["ids"]) for e, _ in tr])
    for ep in range(a.epochs):
        model.train()
        order = np.random.RandomState(41+ep).permutation(len(tr))
        batches, cur, cur_max = [], [], 0
        for oi in order:
            l = lens[oi]
            nm, nn = max(cur_max, l), len(cur)+1
            if cur and (nm*nn > 12288 or nn > 24):
                batches.append(cur); cur, cur_max = [], 0
                nm = l
            cur.append(oi); cur_max = nm
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        t0, tot = time.time(), 0
        for bidx in batches:
            sel = [tr[i][0] for i in bidx]
            klen = [len(tr[i][0]["markers"]) for i in bidx]
            kmax = max(klen)
            tlog = torch.stack([torch.nn.functional.pad(torch.from_numpy(tr[i][1][:klen[j]]),
                                                        (0, kmax-klen[j]), value=-1e4)
                                for j, i in enumerate(bidx)]).to(device)
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
                loss, _ = total_loss(out["choice_logits"], tlog, bd["target"].to(device),
                                     bd["marker_mask"].to(device), bd["qtype"], tau=a.tau)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            tot += loss.item()
        m = evaluate(model, [e for e, _ in va], tok, device, 512, 192)
        print(f"ep {ep+1} loss {tot/max(1,len(batches)):.4f} acc {m['acc']:.3f} brier {m['brier']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    from laya import calibrate as laya_cal
    from laya.common import clamp_temperature
    model.eval()
    records = []
    with torch.no_grad():
        for s in range(0, len(va), 64):
            sel = [e for e, _ in va[s:s+64]]
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
    print("temps:", temps, temps_by, flush=True)
    os.makedirs(f"{a.out}/encoder", exist_ok=True)
    os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
    save_file(model.state_dict(), f"{a.out}/model.safetensors")
    model.encoder.config.save_pretrained(f"{a.out}/encoder")
    tok.save_pretrained(f"{a.out}/tokenizer")
    json.dump({"encoder": "answerdotai/ModernBERT-base", "head_layers": 2, "max_len": 512,
               "head_max_len": 192, "max_prefixes": 6, "act_costs": {"escalate": 0.5},
               "amp_dtype": "fp16", "temperature": temps, "temperature_by_options": temps_by,
               "params_M": round(count_params(model)/1e6,1),
               "teacher": "convaiinnovations/laya soft logits", "tau": a.tau},
              open(f"{a.out}/frontier_config.json","w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
