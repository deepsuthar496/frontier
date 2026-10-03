"""v9: hybrid gold + agree-weighted KD. Selective KD, 4 splits, T4 profile-first.

Loss (supervised baseline + selective KD):
  L = CE(gold) + 0.1*Brier(gold) + 0.1*RPS(ordinal,gold) + alpha*tau^2*KL(teacher||student)
  alpha from KD_POLICY (agree 0.5 / disagree 0.1 / missing 0.0); see common.KD_POLICY.
  Invalid option slots masked before every softmax. RPS only for score.
  Options NEVER shuffled (order locked to teacher/gold).

Splits (data_v8.npz): train (updates), dev (selection), cal (temps only),
final (untouched until a candidate is selected — this script only counts it).

T4 plan: SDPA, fp16+scaler, grad-ckpt, microbatch by length, accumulation to
eff ~32 decisions, encoder LR 5e-6, head LR 2e-5, clip 1.0, curriculum
512 -> 1024. --profile100 first: 100 optimizer steps, print VRAM/dec-s,
extrapolate, exit.
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
import sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file, save_file
from model import FrontierDecisionEngine, count_params
from common import QTYPES, collate, KD_POLICY
from train import encode as encode_rec, evaluate


def load_parts(path="data_v8.npz"):
    z = np.load(path, allow_pickle=True)
    return {k: json.loads(str(z[k])) for k in z.files}


def loss_v8(student_logits, gold_idx, teacher_logits, alpha, mask, qtype, tau=2.0):
    """Gold-first. teacher_logits: [B,K] with -1e4 pads or None."""
    mf = mask.float()
    B, K = student_logits.shape
    gold_oh = torch.zeros_like(student_logits).scatter_(-1, gold_idx[:, None].clamp(0, K - 1), 1.0)
    gold_oh = gold_oh * mf
    # CE against gold (masked softmax)
    lsm = torch.log_softmax(student_logits, -1)
    ce = -(gold_oh * lsm).sum(-1) / mf.sum(-1).clamp_min(1)
    p = torch.softmax(student_logits, -1) * mf
    brier = (((p - gold_oh) ** 2) * mf).sum(-1) / mf.sum(-1).clamp_min(1)
    loss = ce + 0.1 * brier
    is_score = (qtype == QTYPES["score"])
    if is_score.any():
        k = mf.sum(-1).clamp(min=2)
        cdf_q = torch.cumsum(torch.softmax(student_logits, -1) * mf, -1)
        cdf_t = torch.cumsum(gold_oh, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mf).sum(-1) / (k - 1)
        loss = loss + 0.1 * rps * is_score.float()
    klm = torch.zeros((), device=loss.device)
    if teacher_logits is not None:
        tp = torch.softmax(teacher_logits / tau, -1).clamp_min(1e-12)
        kl_elem = F.kl_div(torch.log_softmax(student_logits / tau, -1), tp, reduction="none")
        kl = (kl_elem * mf).sum(-1) * (tau ** 2)
        kl = kl * alpha.to(kl.dtype)
        loss = loss + kl
        klm = kl.mean()
    return loss.mean(), {"ce": ce.mean().item(), "brier": brier.mean().item(), "kl": float(klm.detach())}


def materialize(recs, tok, max_len, head_max_len, rng):
    out = []
    for r in recs:
        q = r["q"]
        k = len(q["crit"]) if q["t"] != "noul" else 2
        soft = [0.0] * k
        if 0 <= int(r["gold"]) < k:
            soft[int(r["gold"])] = 1.0
        e = encode_rec({"state": r["state"],
                        "q": {"t": q["t"], "ins": q["ins"], "crit": q["crit"]},
                        "soft": soft, "y": int(r["gold"])},
                       tok, max_len, head_max_len, False, rng)
        if e is None:
            continue
        t = r.get("teacher")
        tl = np.array(t, dtype=np.float32) if t is not None else None
        out.append((e, tl, float(r.get("alpha", 0.0)), r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="ckpt_in")
    ap.add_argument("--data", default="data_v11.npz")
    ap.add_argument("--variant", default="base", choices=["base", "gold-hard", "selective"],
                    help="KD policy on hard rows (source synth-hard-v1)")
    ap.add_argument("--out", default="frontier_ckpt_v8")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--eff_batch", type=int, default=32)
    ap.add_argument("--profile100", action="store_true")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--init_hf", default=None,
                    help="HF encoder id for fresh init (no local ckpt); warm-starts from Laya")
    ap.add_argument("--teacher_npz", default=None,
                    help="teacher_v9.npz {logits, alphas} aligned to train order; overrides item alpha")
    ap.add_argument("--data_npz", default=None,
                    help="alias: data file (defaults to --data)")
    a = ap.parse_args()
    assert not a.profile100, "profile100 is single-GPU only; profile the base script"
    from datetime import timedelta as _td
    dist.init_process_group("nccl", timeout=_td(minutes=30))  # rank0 eval is long; outlast it
    RANK, WORLD = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(RANK)
    device = torch.device(f"cuda:{RANK}")
    is_main = RANK == 0
    if a.init_hf:
        tok = AutoTokenizer.from_pretrained(a.init_hf)
        enc = AutoModel.from_pretrained(a.init_hf, attn_implementation="sdpa")
        try:
            import laya
            sd = laya.load("convaiinnovations/laya").model.state_dict()
            w = {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}
            missing, unexp = enc.load_state_dict(w, strict=False)
            print(f"Laya warm-start: missing={len(missing)} unexpected={len(unexp)}", flush=True)
        except Exception as e:
            print(f"warm-start skipped ({e})", flush=True)
        model = FrontierDecisionEngine(enc, head_layers=2)
        init_tag = f"{a.init_hf}+laya-warm"
    else:
        tok = AutoTokenizer.from_pretrained(f"{a.src}/tokenizer")
        enc = AutoModel.from_config(
            __import__("transformers").AutoConfig.from_pretrained(f"{a.src}/encoder"),
            attn_implementation="sdpa")
        model = FrontierDecisionEngine(enc, head_layers=json.load(
            open(f"{a.src}/frontier_config.json")).get("head_layers", 2))
        model.load_state_dict(load_file(f"{a.src}/model.safetensors"), strict=True)
        init_tag = a.src
    model.to(device)
    enc.gradient_checkpointing_enable()
    print(f"[rank {RANK}/{WORLD}] params:", round(count_params(model) / 1e6, 1),
          "M from", init_tag, flush=True)

    parts = load_parts(a.data_npz or a.data)
    # variant KD policy on hard rows (spec 3.2); logged per batch below
    _nhard = sum(1 for it in parts["train"] if it.get("source") == "synth-hard-v1")
    for it in parts["train"]:
        if it.get("source") == "synth-hard-v1":
            if a.variant == "gold-hard":
                it["alpha"] = 0.0
            elif a.variant == "selective" and int(it.get("difficulty", 1)) >= 2:
                it["alpha"] = 0.0
    print(f"variant={a.variant} hard_train_rows={_nhard}", flush=True)
    if a.teacher_npz:
        _tz = np.load(a.teacher_npz, allow_pickle=True)
        _TL, _AL = _tz["logits"].astype(np.float32), _tz["alphas"].astype(np.float32)
        assert len(_TL) == len(parts["train"]), (len(_TL), len(parts["train"]))
        from common import content_id as _cid, option_hash as _oph
        from common import KD_POLICY as _KDP
        _meta = json.loads(str(_tz["meta"]))
        assert _meta.get("kd_policy") == _KDP, f"teacher KD policy {_meta.get('kd_policy')} != {_KDP}"
        _ids = [str(x) for x in _tz["ids"]]
        _ophs = [str(x) for x in _tz["option_hash"]]
        _ks = [int(x) for x in _tz["num_options"]]
        for i, it in enumerate(parts["train"]):
            q = {"t": it["q"]["t"], "ins": it["q"]["ins"], "crit": it["q"]["crit"]}
            assert _cid(it["state"], q, it["gold"]) == _ids[i], f"teacher row {i} id mismatch"
            assert _oph(q) == _ophs[i], f"teacher row {i} option mismatch"
            k = len(q["crit"]) if q["t"] != "noul" else 2
            assert _ks[i] == k, f"teacher row {i} k mismatch"
            it["teacher"] = [float(x) for x in _TL[i, :k]]
            it["alpha"] = float(_AL[i])
        nag = int((_AL == _KDP["agree"]).sum())
        print(f"teacher merged+validated: agree{_KDP['agree']}={nag} "
              f"disagree{_KDP['disagree']}={int((_AL==_KDP['disagree']).sum())}", flush=True)
    print({k: len(v) for k, v in parts.items()}, flush=True)
    print(f"final held out: {len(parts['final'])} items (untouched)", flush=True)
    rng = random.Random(8)
    tr = materialize(parts["train"], tok, a.max_len, 192, rng)
    va = materialize(parts["dev"], tok, a.max_len, 192, rng)
    hv = materialize(parts.get("hard-dev", []), tok, a.max_len, 192, rng)
    print(f"train {len(tr)} dev {len(va)} hard-dev {len(hv)} (max_len {a.max_len})", flush=True)

    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 5e-6},
                             {"params": list(model.refine.parameters()) + list(model.choice_head.parameters())
                              + list(model.pool_weights.parameters()) + list(model.score_head.parameters())
                              + list(model.noul_head.parameters()) + list(model.act_head.parameters())
                              + list(model.type_emb.parameters()), "lr": 2e-5}], weight_decay=0.01)
    sched = None  # built after counting real opt-steps (see below)
    scaler = torch.amp.GradScaler("cuda")
    lens = np.array([len(e["ids"]) for e, _, _, _ in tr])
    MICRO = 4
    ACCUM = max(1, a.eff_batch // MICRO // WORLD)  # per-rank; global batch stays eff_batch

    def microbatches_for(ep):
        order = np.random.RandomState(8 + ep).permutation(len(tr))
        # length-bucket for memory, then flat microbatches of MICRO
        buckets, cur, cur_max = [], [], 0
        for oi in order:
            l = lens[oi]
            nm, nn = max(cur_max, l), len(cur) + 1
            if cur and (nm * nn > 16384 or nn > 64):
                buckets.append(cur); cur, cur_max = [], 0
                nm = l
            cur.append(oi); cur_max = nm
        if cur:
            buckets.append(cur)
        rng.shuffle(buckets)
        mbs = []
        for b in buckets:
            for j in range(0, len(b), MICRO):
                mbs.append(b[j:j + MICRO])
        rng.shuffle(mbs)
        return mbs

    steps_per_epoch = None  # counted exactly below via probe
    # exact count: block-strided so both ranks sync together every ACCUM mbs.
    # identical order on both ranks (fixed seeds) -> same global list.
    probe = microbatches_for(0)
    import math as _math
    STRIDE = ACCUM * WORLD
    pad = (-len(probe)) % STRIDE
    opt_per_epoch = (len(probe) + pad) // STRIDE
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * opt_per_epoch)
    if is_main:
        print(f"microbatches/epoch={len(probe)} accum/rank={ACCUM} "
              f"opt-steps/epoch={opt_per_epoch} global_batch={MICRO*ACCUM*WORLD}", flush=True)
    model = DDP(model, device_ids=[RANK], find_unused_parameters=True)  # score/noul/act heads unused by loss_v8

    def my_mbs(mbs):
        mbs = (mbs + mbs[:pad]) if pad else mbs
        mine = []
        for s in range(len(mbs) // STRIDE):
            base = s * STRIDE + RANK * ACCUM
            mine.extend(mbs[base:base + ACCUM])
        return mine

    steps = 0
    t0 = time.time()
    acc_tokens = 0
    for ep in range(a.epochs):
        model.train()
        tot = 0
        nb = 0
        _comp = {}
        _nmb = 0
        mbs = my_mbs(microbatches_for(ep))
        opt.zero_grad()
        for ui, mb in enumerate(mbs):
            sel = [tr[i][0] for i in mb]
            klen = [len(tr[i][0]["markers"]) for i in mb]
            kmax = max(klen)
            rows = []
            for j, i in enumerate(mb):
                if tr[i][1] is not None:
                    v = torch.from_numpy(tr[i][1][:klen[j]])
                else:
                    v = torch.full((klen[j],), -1e4)
                if len(v) < kmax:
                    v = torch.nn.functional.pad(v, (0, kmax - len(v)), value=-1e4)
                rows.append(v)
            has_t = any(tr[i][1] is not None for i in mb)
            tlog = torch.stack(rows).to(device) if has_t else None
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            gold = bd["label"].to(device)
            alpha = torch.tensor([tr[i][2] for i in mb], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"],
                            bd["marker_mask"], bd["qtype"])
                loss, _parts = loss_v8(out["choice_logits"], gold, tlog, alpha,
                                       bd["marker_mask"].to(device), bd["qtype"], tau=a.tau)
                loss = loss / ACCUM
            sync = (ui + 1) % ACCUM == 0  # aligned on both ranks by block stride
            with (nullcontext() if sync else model.no_sync()):
                scaler.scale(loss).backward()
            tot += loss.item()
            _nmb += 1
            for _k, _v in _parts.items():
                _comp[_k] = _comp.get(_k, 0.0) + _v
            _nh = sum(1 for i in mb if tr[i][3].get("source") == "synth-hard-v1")
            _comp["hard_rows"] = _comp.get("hard_rows", 0) + _nh
            _comp["hard_alpha"] = _comp.get("hard_alpha", 0.0) + sum(
                tr[i][2] for i in mb if tr[i][3].get("source") == "synth-hard-v1")
            if sync:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
                steps += 1
                nb += 1
                if a.profile100 and steps >= 100:
                    dt = time.time() - t0
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"PROFILE100: {steps} opt-steps {dt:.0f}s peak={peak:.2f}GB "
                          f"dec/s={steps * a.eff_batch / dt:.1f} toks/s={acc_tokens / dt:.0f}", flush=True)
                    return
            acc_tokens += int(bd["attention_mask"].sum())
        dist.barrier()
        # sharded eval: both ranks work (no 10-min one-sided barrier wait), rank0 aggregates
        def _ev(items):
            shard = [e for j, (e, _, _, _) in enumerate(items) if j % WORLD == RANK]
            ms = evaluate(model, shard, tok, device, a.max_len, 192)
            got = [None for _ in range(WORLD)]
            dist.all_gather_object(got, {"n": len(shard), "acc": ms["acc"], "brier": ms["brier"],
                                         "ece": ms.get("ece"), "score_mae": ms.get("score_mae", 0.0)})
            ntot = sum(g["n"] for g in got)
            return {"acc": sum(g["acc"] * g["n"] for g in got) / max(1, ntot),
                    "brier": sum(g["brier"] * g["n"] for g in got) / max(1, ntot),
                    "ece": got[0]["ece"], "score_mae": got[0]["score_mae"]}
        m = _ev(va)
        mh = _ev(hv) if hv else {"acc": float("nan"), "brier": float("nan"),
                                 "ece": None, "score_mae": 0.0}
        if is_main:
            _PCI = {k: round(v / max(1, _nmb), 4) for k, v in _comp.items()
                    if not k.startswith("hard")}
            _PCIhard = {k: round(v / max(1, _nmb), 4) for k, v in _comp.items() if k.startswith("hard")}
            print(f"ep {ep+1} loss {tot/max(1,nb):.4f} dev acc {m['acc']:.3f} brier {m['brier']:.4f} "
                  f"harddev acc {mh['acc']:.3f} brier {mh['brier']:.4f} parts={_PCI} hard={_PCIhard}", flush=True)
            epdir = f"{a.out}_ep{ep+1}"
            os.makedirs(f"{epdir}/encoder", exist_ok=True)
            os.makedirs(f"{epdir}/tokenizer", exist_ok=True)
            save_file(model.module.state_dict(), f"{epdir}/model.safetensors")
            model.module.encoder.config.save_pretrained(f"{epdir}/encoder")
            tok.save_pretrained(f"{epdir}/tokenizer")
            json.dump({"dev": m, "hard_dev": mh, "ep": ep + 1, "variant": a.variant},
                      open(f"{epdir}/dev_metrics.json", "w"), indent=1)
            print(f"snapshot {epdir}", flush=True)
        dist.barrier()

    # cal split -> global + per-bucket temps (validate buckets help; fallback global)
    from laya import calibrate as laya_cal
    from laya.common import clamp_temperature
    model.eval()
    cal = materialize(parts["cal"], tok, a.max_len, 192, rng)
    records = []
    with torch.no_grad():
        for s in range(0, len(cal), 64):
            sel = [e for e, _, _, _ in cal[s:s + 64]]
            if not sel:
                continue
            b = collate([[it] for it in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"],
                            bd["marker_mask"], bd["qtype"])
            lg = out["choice_logits"].float().cpu().numpy()
            for r, it in enumerate(sel):
                kk = len(it["markers"])
                tgt = np.zeros(kk); tgt[it["label"]] = 1
                records.append((int(it["qtype"]), lg[r, :kk].astype(float), tgt.astype(float), kk))
    fit = laya_cal.fit_temperature_map(records, compute_ece=True, seed=0) if is_main else None
    if is_main:
        temps = [clamp_temperature(t) for t in fit["temperature"]]
        temps_by = {k: clamp_temperature(v) for k, v in fit.get("temperature_by_options", {}).items()}
        print("temps:", temps, temps_by, flush=True)
        os.makedirs(f"{a.out}/encoder", exist_ok=True)
        os.makedirs(f"{a.out}/tokenizer", exist_ok=True)
        save_file(model.module.state_dict(), f"{a.out}/model.safetensors")
        fp16_sd = {k: (v.half() if v.is_floating_point() else v)
                   for k, v in model.module.state_dict().items()}
        save_file(fp16_sd, f"{a.out}/model_fp16.safetensors")
        model.module.encoder.config.save_pretrained(f"{a.out}/encoder")
        tok.save_pretrained(f"{a.out}/tokenizer")
        json.dump({"encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": a.max_len,
                   "head_max_len": 192, "temperature": temps, "temperature_by_options": temps_by,
                   "params_M": round(count_params(model) / 1e6, 1), "teacher": "selective-KD agree-only",
                   "tau": a.tau, "init": a.src, "world": WORLD,
                   "kd_policy": KD_POLICY},
                  open(f"{a.out}/frontier_config.json", "w"), indent=1)
        print("saved", a.out, f"final split still untouched: {len(parts['final'])}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
