"""v7-large: size restriction lifted. ModernBERT-large (395M) + Frontier head.

Why this scales without a perf bottleneck:
- Same single-pass marker scheme as v3/v6 (1 encoder forward). engine_t4's
  dual-encoder does 2 forwards; this does 1, so latency tracks Laya (~30-40ms),
  not 2x. Encoder 22->28 layers is ~+30% FLOPs, still T4-friendly in fp16/SDPA.
- Warm-start encoder from Laya's fine-tuned ModernBERT-large (strip `encoder.`
  prefix; HF keys match). Student starts at teacher's representation, KL then
  refines the head instead of relearning language from scratch.
- Teacher: same teacher_labels.npz Laya soft logits (KL) + CE + Brier + RPS.

Usage:
  python train_v7_large.py --epochs 2 --out frontier_ckpt_v7
  python bench_tdb.py --ckpt frontier_ckpt_v7  (promote only if > 61.83 half)
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
from train import encode as encode_rec, evaluate


def load_teacher(path="teacher_labels.npz"):
    z = np.load(path, allow_pickle=True)
    recs = json.loads(str(z["recs"]))
    return recs, z["teacher_logits"].astype(np.float32)


def load_laya_encoder_weights(device="cpu"):
    """Return state_dict of Laya's ModernBERT-large encoder with HF key names."""
    import laya
    agent = laya.load("convaiinnovations/laya")
    sd = agent.model.state_dict()
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    del agent
    import gc; gc.collect()
    if device == "cpu":
        enc = {k: v.cpu() for k, v in enc.items()}
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_tok", default="frontier_ckpt_v3/tokenizer")
    ap.add_argument("--encoder_name", default="answerdotai/ModernBERT-large")
    ap.add_argument("--out", default="frontier_ckpt_v7")
    ap.add_argument("--teacher", default="teacher_labels.npz")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--head_layers", type=int, default=2)
    ap.add_argument("--no_warm", action="store_true", help="skip Laya encoder warm-start")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--head_max_len", type=int, default=192)
    a = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.src_tok)
    enc = AutoModel.from_pretrained(a.encoder_name, attn_implementation="sdpa")
    if not a.no_warm:
        try:
            w = load_laya_encoder_weights()
            missing, unexpected = enc.load_state_dict(w, strict=False)
            print(f"warm-start Laya encoder: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        except Exception as e:
            print(f"warm-start failed ({e}); using HF pretrained", flush=True)
    model = FrontierDecisionEngine(enc, head_layers=a.head_layers)
    model.to(device)
    enc.gradient_checkpointing_enable()
    print("params:", round(count_params(model)/1e6, 1), "M", flush=True)

    recs, TL = load_teacher(a.teacher)
    print(f"teacher labels: {len(recs)} records", flush=True)
    rng = random.Random(41)
    items = []
    for i, r in enumerate(recs):
        q = r["q"]
        e = encode_rec({"state": r["state"], "q": {"t": q["t"], "ins": q["ins"], "crit": q["crit"]},
                        "soft": r["soft"], "y": r["y"]},
                       tok, a.max_len, a.head_max_len, False, rng)
        if e is None:
            continue
        items.append((e, TL[i]))
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(len(idx)*0.9)
    tr = [items[i] for i in idx[:cut]]
    va = [items[i] for i in idx[cut:]]
    print(f"train {len(tr)} val {len(va)} (dropped {len(recs)-len(items)})", flush=True)

    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-5},
                             {"params": list(model.refine.parameters()) + list(model.choice_head.parameters())
                              + list(model.pool_weights.parameters()) + list(model.score_head.parameters())
                              + list(model.noul_head.parameters()) + list(model.act_head.parameters())
                              + list(model.type_emb.parameters()), "lr": 5e-5}],
                            weight_decay=0.01)
    # smaller token budget for large encoder on T4
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*max(1, len(tr)//12))
    scaler = torch.amp.GradScaler("cuda")
    lens = np.array([len(e["ids"]) for e, _ in tr])
    for ep in range(a.epochs):
        model.train()
        order = np.random.RandomState(41+ep).permutation(len(tr))
        batches, cur, cur_max = [], [], 0
        for oi in order:
            l = lens[oi]
            nm, nn = max(cur_max, l), len(cur)+1
            if cur and (nm*nn > 8192 or nn > 12):
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
        m = evaluate(model, [e for e, _ in va], tok, device, a.max_len, a.head_max_len)
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
    from safetensors.torch import save_file
    os.makedirs(f"{a.out}/encoder", exist_ok=True)
    os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
    save_file(model.state_dict(), f"{a.out}/model.safetensors")
    model.encoder.config.save_pretrained(f"{a.out}/encoder")
    tok.save_pretrained(f"{a.out}/tokenizer")
    json.dump({"encoder": a.encoder_name, "head_layers": a.head_layers, "max_len": a.max_len,
               "head_max_len": a.head_max_len, "max_prefixes": 6, "act_costs": {"escalate": 0.5},
               "amp_dtype": "fp16", "temperature": temps, "temperature_by_options": temps_by,
               "params_M": round(count_params(model)/1e6, 1),
               "teacher": "convaiinnovations/laya soft logits", "tau": a.tau,
               "warm_start": (not a.no_warm)},
              open(f"{a.out}/frontier_config.json", "w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
