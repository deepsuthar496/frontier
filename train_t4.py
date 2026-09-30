"""Train SystemOneDecisionEngine (engine_t4 spec) to frontier quality on T4.
Local-only work. No Drive sync here — backup only when metrics are good.

Data: synth_data.build_records (same generator as train.py, no API keys).
I/O (spec): doc = state text; options = rendered option strings, CLS-pooled.
Loss (spec s4): CE + tau^2*KL(soft||student) + Brier. Aux: score/noul heads.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F

from engine_t4 import SystemOneDecisionEngine, count_params
from synth_data import build_records
from common import render_options


def encode_record(rec):
    q = rec["q"]
    opts = render_options(q)
    return {"state": rec["state"], "options": opts,
            "soft": list(rec["soft"]), "label": int(rec["y"]),
            "t": q["t"]}


def collate_t4(items, tok, max_len=512, max_opt_len=64):
    docs = tok([it["state"] for it in items], padding=True, truncation=True,
               max_length=max_len, return_tensors="pt")
    K = max(len(it["options"]) for it in items)
    opt_ids, opt_mask = [], []
    for it in items:
        t = tok(it["options"], padding="max_length", truncation=True,
                max_length=max_opt_len, return_tensors="pt")
        pad_n = K - len(it["options"])
        if pad_n > 0:
            pad_id = tok.pad_token_id
            t_ids = torch.cat([t["input_ids"],
                               torch.full((pad_n, max_opt_len), pad_id, dtype=torch.long)], 0)
            t_m = torch.cat([t["attention_mask"],
                             torch.zeros((pad_n, max_opt_len), dtype=torch.long)], 0)
        else:
            t_ids, t_m = t["input_ids"], t["attention_mask"]
        opt_ids.append(t_ids); opt_mask.append(t_m)
    opt_ids = torch.stack(opt_ids)   # [B,K,Lo]
    opt_mask = torch.stack(opt_mask)
    Kmask = torch.zeros(len(items), K, dtype=torch.bool)
    tgt = torch.zeros(len(items), K)
    for i, it in enumerate(items):
        k = len(it["options"])
        Kmask[i, :k] = True
        tgt[i, :k] = torch.tensor(it["soft"], dtype=torch.float32)
    labs = torch.tensor([it["label"] for it in items], dtype=torch.long)
    return {"input_ids": docs["input_ids"], "attention_mask": docs["attention_mask"],
            "option_ids": opt_ids, "option_mask": opt_mask,
            "marker_mask": Kmask, "target": tgt, "label": labs,
            "t": [it["t"] for it in items]}


def loss_fn(student_logits, target, mask, tau=2.0, l_ce=1.0, l_kl=1.0, l_brier=0.5):
    mf = mask.float()
    logp = torch.log_softmax(student_logits, -1)
    ce = -(target * logp * mf).sum(-1) / mf.sum(-1).clamp_min(1)
    tp = torch.softmax((torch.log(target.clamp_min(1e-6))) / tau, -1) * mf
    tp = tp / tp.sum(-1, keepdim=True).clamp_min(1e-9)
    kl = F.kl_div(torch.log_softmax(student_logits / tau, -1), tp,
                  reduction="none").sum(-1) * (tau ** 2)
    p = torch.softmax(student_logits, -1) * mf
    brier = (((p - target * mf) ** 2) * mf).sum(-1) / mf.sum(-1).clamp_min(1)
    return (l_ce * ce + l_kl * kl + l_brier * brier).mean(), ce.mean().item()


@torch.no_grad()
def evaluate(model, items, tok, device):
    model.eval()
    corr, brier, conf_c, conf_t = [], [], [], []
    for s in range(0, len(items), 16):
        sel = items[s:s + 16]
        b = collate_t4(sel, tok)
        bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in b.items() if k not in ("t",)}
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(bd["input_ids"], bd["attention_mask"],
                        bd["option_ids"], bd["option_mask"])
        lg = out["choice_logits"].float().cpu().numpy()
        mm = b["marker_mask"].numpy()
        for r, it in enumerate(sel):
            k = int(mm[r].sum())
            z = lg[r, :k]
            e = np.exp(z - z.max()); p = e / e.sum()
            pred = int(p.argmax())
            corr.append(pred == it["label"])
            tgt = np.zeros(k); tgt[it["label"]] = 1
            brier.append(float(((p - tgt) ** 2).mean()))
            conf_c.append(float(p.max())); conf_t.append(pred == it["label"])
    acc = float(np.mean(corr))
    # simple ECE on max-prob
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    cc, ct = np.array(conf_c), np.array(conf_t)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (cc > lo) & (cc <= hi)
        if sel.any():
            ece += sel.mean() * abs(cc[sel].mean() - ct[sel].mean())
    return {"acc": acc, "brier": float(np.mean(brier)), "ece": float(ece), "n": len(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frontier_ckpt_t4")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--n_train", type=int, default=2400)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device,
          torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(a.encoder)
    enc = AutoModel.from_pretrained(a.encoder, attn_implementation="sdpa")
    if device.type == "cuda":
        enc.gradient_checkpointing_enable()
    model = SystemOneDecisionEngine(enc).to(device)
    print(f"params: {count_params(model)/1e6:.1f}M (spec target ~154M)")

    recs = build_records(n_per_workflow=(a.n_train + a.n_val) // 4, seed=a.seed)
    items = [encode_record(r) for r in recs]
    random.Random(a.seed).shuffle(items)
    tr, va = items[:a.n_train], items[a.n_train:a.n_train + a.n_val]
    print(f"train {len(tr)} val {len(va)}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs * 100))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for ep in range(a.epochs):
        model.train()
        idx = np.random.RandomState(a.seed + ep).permutation(len(tr))
        tot, t0 = 0.0, time.time()
        for s in range(0, len(tr), 8):
            sel = [tr[i] for i in idx[s:s + 8]]
            b = collate_t4(sel, tok)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items() if k not in ("t",)}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = model(bd["input_ids"], bd["attention_mask"],
                            bd["option_ids"], bd["option_mask"])
                loss, _ = loss_fn(out["choice_logits"], bd["target"].to(out["choice_logits"].dtype),
                                  bd["marker_mask"])
                # aux heads: score->normalized label, noul->BCE(label==1 weighted)
                aux = 0
                is_score = torch.tensor([t == "score" for t in b["t"]], device=device)
                if is_score.any():
                    k = bd["marker_mask"].float().sum(-1).clamp_min(2)
                    norm = bd["label"].float() / (k - 1)
                    aux = aux + F.mse_loss(out["score"][is_score].squeeze(-1),
                                           norm[is_score].to(out["score"].dtype))
                is_noul = torch.tensor([t == "noul" for t in b["t"]], device=device)
                if is_noul.any():
                    aux = aux + F.binary_cross_entropy_with_logits(
                        out["noul_logit"][is_noul].squeeze(-1) if out["noul_logit"].dim() > 1
                        else out["noul_logit"][is_noul],
                        (bd["label"][is_noul] == 1).float().to(out["noul_logit"].dtype))
                loss = loss + 0.3 * aux
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step(); tot += loss.item()
        m = evaluate(model, va, tok, device)
        print(f"epoch {ep+1}/{a.epochs} loss {tot/max(1,len(tr)//8):.4f} "
              f"val acc {m['acc']:.3f} brier {m['brier']:.4f} ece {m['ece']:.3f} "
              f"({time.time()-t0:.0f}s)")

    m = evaluate(model, va, tok, device)
    print("FINAL:", m)
    # fit single temperature on val NLL (grid)
    model.eval()
    best_t, best_nll = 1.0, 1e18
    with torch.no_grad():
        allz, ally = [], []
        for s in range(0, len(va), 16):
            sel = va[s:s + 16]
            b = collate_t4(sel, tok)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items() if k not in ("t",)}
            out = model(bd["input_ids"], bd["attention_mask"],
                        bd["option_ids"], bd["option_mask"])
            lg = out["choice_logits"].float().cpu().numpy()
            mm = b["marker_mask"].numpy()
            for r, it in enumerate(sel):
                k = int(mm[r].sum()); allz.append(lg[r, :k]); ally.append(it["label"])
    for t in [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]:
        nlls = []
        for z, y in zip(allz, ally):
            zt = z / t
            e = np.exp(zt - zt.max()); p = e / e.sum()
            nlls.append(-np.log(max(p[y], 1e-9)))
        nll = float(np.mean(nlls))
        if nll < best_nll: best_nll, best_t = nll, t
    print("temperature:", best_t, "nll", round(best_nll, 4))
    model.temperature.data.fill_(best_t)

    os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), f"{a.out}/model.safetensors")
    model.encoder.config.save_pretrained(f"{a.out}/encoder")
    tok.save_pretrained(f"{a.out}/tokenizer")
    with open(f"{a.out}/frontier_config.json", "w") as f:
        json.dump({"encoder": a.encoder, "arch": "SystemOneDecisionEngine",
                   "max_len": 512, "temperature": float(best_t),
                   "params_M": round(count_params(model) / 1e6, 1),
                   "val": m}, f, indent=2)
    print(f"saved -> {a.out} (LOCAL ONLY, not synced to Drive)")


if __name__ == "__main__":
    main()
