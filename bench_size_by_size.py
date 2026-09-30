"""Size-by-size bench: Frontier vs Laya, like-for-like.

Correctness notes (from laya source):
- Calibration figures use answer_confidence = max(p), NOT entropy confidence
  (laya/common.py:510, laya/confidence.py:10). ECE via laya.common.ece_score (15 bins).
- Temperatures must be in [0.5, 5.0] (laya/common.py:552 TEMP_MIN/MAX); per-bucket
  fits need MIN_BUCKET_N=2000 (laya/calibrate.py) — small samples get type scalar only.
- Same inputs, same fp16/autocast harness, warmup + cuda sync, end-to-end predict().
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")

from laya.common import ece_score, answer_confidence, clamp_temperature  # noqa: E402
from laya import calibrate as laya_cal  # noqa: E402

FRONTIER_DIR = "/teamspace/studios/this_studio/frontier/frontier_ckpt_final"
DEVICE = "cuda"


def section(name):
    print(f"\n===== {name} =====", flush=True)


def size_table():
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel
    sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
    from model import FrontierDecisionEngine
    # frontier
    ecfg = AutoConfig.from_pretrained(f"{FRONTIER_DIR}/encoder")
    enc = AutoModel.from_config(ecfg)
    fm = FrontierDecisionEngine(enc, head_layers=2)
    fm.load_state_dict(load_file(f"{FRONTIER_DIR}/model.safetensors"), strict=True)
    f_total = sum(p.numel() for p in fm.parameters())
    f_enc = sum(p.numel() for p in fm.encoder.parameters())
    fcfg = json.load(open(f"{FRONTIER_DIR}/frontier_config.json"))
    # laya (live checkpoint, meta-device build to avoid double VRAM)
    import laya
    agent = laya.load("convaiinnovations/laya")
    m = agent.model
    l_total = sum(p.numel() for p in m.parameters())
    l_enc = sum(p.numel() for p in m.encoder.parameters())
    del m
    rows = {
        "frontier": {"total_M": round(f_total/1e6,1), "encoder_M": round(f_enc/1e6,1),
                     "head_M": round((f_total-f_enc)/1e6,2),
                     "safetensors_MB": round(os.path.getsize(f"{FRONTIER_DIR}/model.safetensors")/1e6,1),
                     "max_len": fcfg["max_len"], "head_max_len": fcfg["head_max_len"],
                     "temperature": fcfg["temperature"], "temperature_by_options": fcfg.get("temperature_by_options",{}),
                     "temps_clamped_ok": all(0.5 <= clamp_temperature(t) == t for t in fcfg["temperature"])},
        "laya_en": {"total_M": round(l_total/1e6,1), "encoder_M": round(l_enc/1e6,1),
                    "head_M": round((l_total-l_enc)/1e6,2),
                    "max_len": agent.cfg["max_len"], "head_max_len": agent.cfg["head_max_len"],
                    "temperature": agent.cfg.get("temperature"),
                    "temperature_by_options": agent.cfg.get("temperature_by_options", {})},
    }
    print(json.dumps(rows, indent=1))
    return rows


def speed_table():
    import laya
    from agent import FrontierAgent
    la = laya.load("convaiinnovations/laya")
    fa = FrontierAgent(FRONTIER_DIR)
    state_short = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    state_long = ("Invoice #4411 dispute. " + "The March statement shows a duplicate charge of $412.77. " * 90
                  + "Customer requests same-day refund or cancellation.")
    crit = {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts", "other": "everything else"}
    out = {}
    for name, state in [("short", state_short), ("long", state_long)]:
        for nq in [1, 5, 10]:
            for tag, fn in [("laya", lambda: la.predict(state, {f"q{i}": {"type": "choice", "instructions": "Which department handles this?", "criteria": crit} for i in range(nq)})),
                            ("frontier", lambda: fa.system_one(state, {f"q{i}": {"type": "choice", "instructions": "Which department handles this?", "criteria": crit} for i in range(nq)}))]:
                for _ in range(3):
                    fn()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ts = []
                reps = 20 if name == "short" else 10
                for _ in range(reps):
                    t0 = time.perf_counter(); fn()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    ts.append((time.perf_counter()-t0)*1000)
                out[f"{tag}/{name}/{nq}q"] = {"p50_ms": round(float(np.median(ts)),1),
                                             "p99_ms": round(float(np.percentile(ts,99)),1),
                                             "ms_per_q": round(float(np.median(ts))/nq,2)}
                print(f"{tag}/{name}/{nq}q: {out[f'{tag}/{name}/{nq}q']}", flush=True)
    return out


def load_public():
    from datasets import load_dataset
    try:
        ag = load_dataset("fancyzhx/ag_news", split="test[:200]")
    except Exception:
        ag = load_dataset("sh0416/ag_news", split="test[:200]")
    em = load_dataset("dair-ai/emotion", split="test[:150]")
    return ag, em


def predict_probs(agent_fn, state, qdef):
    """Return prob vector in option order + raw logits if available."""
    r = agent_fn(state, {"q": qdef})
    a = r["answers"]["q"]
    if a["type"] == "choice":
        keys = list(qdef["criteria"].keys())
        return np.array([a["probabilities"][k] for k in keys]), keys
    raise ValueError(a)


def accuracy_table():
    import laya
    from agent import FrontierAgent
    la = laya.load("convaiinnovations/laya")
    fa = FrontierAgent(FRONTIER_DIR)
    ag, em = load_public()
    ag_labels = ["World", "Sports", "Business", "Sci/Tech"]
    em_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    suites = {
        "ag_news": (ag, "text", "label", ag_labels, "Which news topic is this article?"),
        "emotion": (em, "text", "label", em_labels, "Which emotion does this text express?"),
    }
    out = {}
    for sname, (ds, tcol, lcol, labels, ins) in suites.items():
        for tag, fn in [("laya", la.predict), ("frontier", fa.system_one)]:
            correct, confs, brier, nll, corrs = [], [], [], [], []
            qdef = {"type": "choice", "instructions": ins, "criteria": {l: l for l in labels}}
            for row in ds:
                try:
                    p, keys = predict_probs(fn, row[tcol][:2000], qdef)
                except Exception:
                    continue
                y = int(row[lcol])
                pred = int(p.argmax())
                ok = pred == y
                correct.append(ok); corrs.append(ok)
                confs.append(answer_confidence(p, len(p)))
                tgt = np.zeros(len(p)); tgt[y] = 1
                brier.append(float(((p-tgt)**2).mean()))
                nll.append(float(-np.log(max(p[y], 1e-9))))
            acc = float(np.mean(correct))
            out[f"{tag}/{sname}"] = {"n": len(correct), "acc": round(acc,4),
                                     "ece_maxp": round(ece_score(np.array(confs), np.array(corrs)),4),
                                     "brier": round(float(np.mean(brier)),4),
                                     "nll": round(float(np.mean(nll)),4),
                                     "mean_maxconf": round(float(np.mean(confs)),4)}
            print(f"{tag}/{sname}: {out[f'{tag}/{sname}']}", flush=True)
    # option-order robustness: 60 AG News cases, shuffled criteria order
    import random
    rng = random.Random(0)
    flips = {}
    rows = [r for r in ag][:60]
    for tag, fn in [("laya", la.predict), ("frontier", fa.system_one)]:
        fl = 0; n = 0
        for row in rows:
            labs = ag_labels[:]
            q1 = {"type": "choice", "instructions": "Which news topic is this article?", "criteria": {l: l for l in labs}}
            labs2 = labs[:]; rng.shuffle(labs2)
            q2 = {"type": "choice", "instructions": "Which news topic is this article?", "criteria": {l: l for l in labs2}}
            try:
                p1, _ = predict_probs(fn, row["text"][:2000], q1)
                p2, k2 = predict_probs(fn, row["text"][:2000], q2)
                a1 = labs[int(p1.argmax())]
                a2 = k2[int(p2.argmax())]
                fl += (a1 != a2); n += 1
            except Exception:
                continue
        flips[tag] = {"n": n, "flip_rate": round(fl/max(n,1),4)}
    print("order flips:", flips)
    out["order_flip"] = flips
    return out


if __name__ == "__main__":
    section("1. SIZE")
    size = size_table()
    section("2. SPEED (T4, fp16, end-to-end predict)")
    speed = speed_table()
    section("3. ACCURACY zero-shot public (AG News 200, Emotion 150) + order robustness")
    acc = accuracy_table()
    json.dump({"size": size, "speed": speed, "accuracy": acc},
              open("/teamspace/studios/this_studio/frontier/bench_results.json","w"), indent=1)
    print("\nSaved frontier/bench_results.json")
