"""T4 training loop: CE + KL-distill + Brier + RPS, fp16, grad-checkpoint, token-budget batching."""
import argparse, json, os, random, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from common import QTYPES, build_sequence, collate, total_loss, ece_score, confidence_from_probs, temp_bucket
from model import FrontierDecisionEngine, count_params
from synth_data import build_records


def encode(rec, tok, max_len, head_max_len, train, rng):
    from common import render_options
    q = rec["q"]; k = len(render_options(q))
    order = list(range(k))
    if train and q["t"] != "score":
        rng.shuffle(order)
    ids, markers = build_sequence(tok, rec["state"], q, max_len, head_max_len, option_order=order)
    if len(markers) != k:
        return None
    soft = [rec["soft"][i] for i in order]
    y = order.index(rec["y"])
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]],
            "target": soft, "label": y, "qtype_name": q["t"], "k": k}


def make_batches(items, max_tokens=12288, max_seqs=24, seed=0):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    lens = np.array([len(items[i]["ids"]) for i in order])
    idx = np.argsort(lens)  # length-sort within shuffle for T4 efficiency
    order = order[idx].tolist()
    batches, cur, cur_max = [], [], 0
    for i in order:
        l = len(items[i]["ids"])
        nm, nn = max(cur_max, l), len(cur) + 1
        if cur and (nm * nn > max_tokens or nn > max_seqs):
            batches.append(cur); cur, cur_max = [], 0
            nm = l
        cur.append(i); cur_max = nm
    if cur: batches.append(cur)
    rng.shuffle(batches)
    return batches


def evaluate(model, items, tok, device, max_len, head_max_len):
    model.eval()
    logs, labs, confs, corr, brier, maes = [], [], [], [], [], []
    tot_loss = 0; n = 0
    with torch.no_grad():
        for s in range(0, len(items), 64):
            sel = items[s:s+64]
            b = collate([[it] for it in sel], tok.pad_token_id)
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                out = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
            lg = out["choice_logits"].float().cpu().numpy()
            for r, it in enumerate(sel):
                kk = len(it["markers"])
                z = lg[r, :kk]
                p = np.exp(z - z.max()); p /= p.sum()
                pred = int(p.argmax())
                logs.append(z); labs.append(it["label"])
                c = confidence_from_probs(p, kk); confs.append(c); corr.append(pred == it["label"])
                tgt = np.zeros(kk); tgt[it["label"]] = 1
                brier.append(float(((p - tgt) ** 2).mean()))
                if it["qtype_name"] == "score":
                    maes.append(abs(float((np.arange(kk)*p).sum()) - it["label"]))
    logs = np.array([np.pad(l, (0, 0)) if len(l)==max(len(x) for x in logs) else l for l in logs], dtype=object)
    acc = float(np.mean(corr))
    return {"acc": acc, "ece": ece_score(np.array(confs), np.array(corr)),
            "brier": float(np.mean(brier)),
            "score_mae": float(np.mean(maes)) if maes else 0.0}


def fit_temperatures(model, items, tok, device):
    """Per-bucket temperature via grid search on NLL (LBFGS-equivalent, CPU-light for T4)."""
    from collections import defaultdict
    model.eval()
    bucket_logits = defaultdict(list); bucket_labels = defaultdict(list)
    with torch.no_grad():
        for s in range(0, len(items), 64):
            sel = items[s:s+64]
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                out = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
            lg = out["choice_logits"].float().cpu().numpy()
            for r, it in enumerate(sel):
                kk = len(it["markers"])
                bucket_logits[temp_bucket(it["qtype"], kk)].append(lg[r, :kk])
                bucket_labels[temp_bucket(it["qtype"], kk)].append(it["label"])
    temps, temps_by = [1.0, 1.0, 1.0], {}
    for bk, arr in bucket_logits.items():
        L = np.stack([np.pad(a, (0, 0)) for a in arr]) if len({len(a) for a in arr})==1 else arr
        best_t, best_nll = 1.0, 1e18
        for t in [0.3,0.5,0.7,1.0,1.3,1.6,2.0,2.5,3.0]:
            nll = 0
            for z, y in zip(arr, bucket_labels[bk]):
                zz = z / t; e = np.exp(zz-zz.max()); p = e/e.sum()
                nll += -np.log(max(p[y],1e-9))
            nll /= len(arr)
            if nll < best_nll: best_nll, best_t = nll, t
        temps_by[bk] = float(best_t)
        qt = QTYPES[bk.split(":")[0]]
        temps[qt] = float(np.mean([v for k2,v in temps_by.items() if k2.startswith(bk.split(':')[0])]))
    return temps, temps_by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frontier_ckpt")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--head_max_len", type=int, default=192)
    ap.add_argument("--n_train", type=int, default=2400)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")
    tok = AutoTokenizer.from_pretrained(a.encoder)
    enc = AutoModel.from_pretrained(a.encoder, attn_implementation="sdpa")
    if device.type == "cuda":  # T4: grad checkpoint encoder to fit
        enc.gradient_checkpointing_enable()
        enc.config.gradient_checkpointing = True
    model = FrontierDecisionEngine(enc, head_layers=2).to(device)
    print(f"params: {count_params(model)/1e6:.1f}M (Laya 421M)")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*100)

    recs = build_records(n_per_workflow=(a.n_train+a.n_val)//4, seed=a.seed)
    rng = random.Random(a.seed)
    items = [e for r in recs for e in [encode(r, tok, a.max_len, a.head_max_len, True, rng)] if e]
    cut = int(len(items)*a.n_train/(a.n_train+a.n_val))
    tr, va = items[:cut], items[cut:cut+a.n_val*2][:a.n_val]
    print(f"train {len(tr)} val {len(va)}")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    step = 0
    for ep in range(a.epochs):
        model.train()
        batches = make_batches(tr, seed=a.seed+ep)
        tot = 0
        t0 = time.time()
        for bi, bidx in enumerate(batches):
            sel = [tr[i] for i in bidx]
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            teacher = (torch.log(bd["target"].clamp_min(1e-6))).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                out = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"], bd["marker_mask"], bd["qtype"])
                loss, parts = total_loss(out["choice_logits"], teacher, bd["target"].to(device),
                                         bd["marker_mask"].to(device), bd["qtype"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step(); tot += loss.item(); step += 1
        m = evaluate(model, va, tok, device, a.max_len, a.head_max_len)
        print(f"epoch {ep+1}/{a.epochs} loss {tot/max(1,len(batches)):.4f} "
              f"val acc {m['acc']:.3f} ece {m['ece']:.3f} brier {m['brier']:.4f} "
              f"score_mae {m['score_mae']:.3f} ({time.time()-t0:.0f}s)")

    temps, temps_by = fit_temperatures(model, va, tok, device)
    print("temperatures:", temps, temps_by)
    model.temperature = torch.nn.Parameter(torch.tensor(temps)) if isinstance(model.temperature, torch.Tensor) else model.temperature
    # save (safetensors + encoder/tokenizer/config for offline load)
    os.makedirs(f"{a.out}/encoder", exist_ok=True)
    os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), f"{a.out}/model.safetensors")
    model.encoder.config.save_pretrained(f"{a.out}/encoder")
    tok.save_pretrained(f"{a.out}/tokenizer")
    with open(f"{a.out}/frontier_config.json", "w") as f:
        json.dump({"encoder": a.encoder, "head_layers": 2, "max_len": 8192 if False else a.max_len,
                   "head_max_len": a.head_max_len, "max_prefixes": 6,
                   "act_costs": {"escalate": 0.5}, "amp_dtype": "fp16",
                   "temperature": temps, "temperature_by_options": temps_by,
                   "params_M": round(count_params(model)/1e6,1)}, f, indent=2)
    m = evaluate(model, va, tok, device, a.max_len, a.head_max_len)
    print("FINAL:", m)


if __name__ == "__main__":
    main()
