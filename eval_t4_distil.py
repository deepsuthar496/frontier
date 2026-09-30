"""Honest public eval for frontier_ckpt_t4 (local only).
Same methodology as bench_size_by_size.py: max(p) confidence, ECE 15 bins,
AG News 200 + Emotion 150 + 60-case order-flip. Compares vs Laya teacher.
"""
import sys, random
sys.path.insert(0, "/content/frontier")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from engine_t4 import SystemOneDecisionEngine
from laya.common import ece_score, answer_confidence


def load_t4(dir="frontier_ckpt_t4_distil", device="cuda"):
    tok = AutoTokenizer.from_pretrained(f"{dir}/tokenizer")
    enc = AutoModel.from_pretrained("answerdotai/ModernBERT-base",
                                    attn_implementation="sdpa")
    m = SystemOneDecisionEngine(enc).to(device)
    m.load_state_dict(load_file(f"{dir}/model.safetensors"), strict=True)
    m.eval()
    return m, tok


@torch.no_grad()
def t4_probs(m, tok, state, labels, device="cuda"):
    d = tok([state[:2000]], padding=True, truncation=True,
            max_length=512, return_tensors="pt").to(device)
    o = tok(labels, padding=True, truncation=True,
            max_length=64, return_tensors="pt").to(device)
    with torch.autocast(device_type=device, dtype=torch.float16,
                        enabled=device == "cuda"):
        out = m(d["input_ids"], d["attention_mask"],
                o["input_ids"], o["attention_mask"])
    return out["choice_probs"].float().cpu().numpy()[0]


def main():
    import laya
    from datasets import load_dataset
    device = "cuda"
    m, tok = load_t4(device=device)
    la = laya.load("convaiinnovations/laya")
    try:
        ag = load_dataset("fancyzhx/ag_news", split="test[:200]")
    except Exception:
        ag = load_dataset("sh0416/ag_news", split="test[:200]")
    em = load_dataset("dair-ai/emotion", split="test[:150]")
    ag_labels = ["World", "Sports", "Business", "Sci/Tech"]
    em_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for sname, ds, tcol, lcol, labels in [
            ("ag_news", ag, "text", "label", ag_labels),
            ("emotion", em, "text", "label", em_labels)]:
        for tag in ["laya", "t4"]:
            ok, confs, corr, brier, nll = [], [], [], [], []
            for row in ds:
                y = int(row[lcol])
                try:
                    if tag == "laya":
                        r = la.predict(row[tcol][:2000], {"q": {
                            "type": "choice", "instructions": "Which?",
                            "criteria": {l: l for l in labels}}})
                        p = np.array([r["answers"]["q"]["probabilities"][l]
                                      for l in labels])
                    else:
                        p = t4_probs(m, tok, row[tcol], labels)
                except Exception:
                    continue
                pred = int(p.argmax())
                c = pred == y
                ok.append(c); corr.append(c)
                confs.append(answer_confidence(p, len(p)))
                tgt = np.zeros(len(p)); tgt[y] = 1
                brier.append(float(((p - tgt) ** 2).mean()))
                nll.append(float(-np.log(max(p[y], 1e-9))))
            print(f"{tag}/{sname}: n={len(ok)} acc={np.mean(ok):.4f} "
                  f"ece={ece_score(np.array(confs), np.array(corr)):.4f} "
                  f"brier={np.mean(brier):.4f} nll={np.mean(nll):.4f} "
                  f"meanconf={np.mean(confs):.4f}", flush=True)
    rng = random.Random(0)
    rows = [r for r in ag][:60]
    for tag in ["laya", "t4"]:
        fl = n = 0
        for row in rows:
            l2 = ag_labels[:]; rng.shuffle(l2)
            try:
                if tag == "laya":
                    r1 = la.predict(row["text"][:2000], {"q": {
                        "type": "choice", "instructions": "Which?",
                        "criteria": {l: l for l in ag_labels}}})
                    r2 = la.predict(row["text"][:2000], {"q": {
                        "type": "choice", "instructions": "Which?",
                        "criteria": {l: l for l in l2}}})
                    a1 = ag_labels[int(np.argmax([r1["answers"]["q"]["probabilities"][l] for l in ag_labels]))]
                    a2 = l2[int(np.argmax([r2["answers"]["q"]["probabilities"][l] for l in l2]))]
                else:
                    p1 = t4_probs(m, tok, row["text"], ag_labels)
                    p2 = t4_probs(m, tok, row["text"], l2)
                    a1 = ag_labels[int(p1.argmax())]; a2 = l2[int(p2.argmax())]
                fl += (a1 != a2); n += 1
            except Exception:
                continue
        print(f"order_flip {tag}: {fl}/{n} = {fl/max(n,1):.4f}", flush=True)


if __name__ == "__main__":
    main()
