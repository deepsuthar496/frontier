"""Distill Laya (Apache-2.0, local teacher) into frontier_ckpt_t4. Local only.
Research basis: Laya card — base 0.362 -> 0.766 via fine-tuning on domain
decisions; RLCD proper-scoring; per-bucket temps. BLOCKERS.md #1 unlock.
Teacher-labels AG-News-train + synth states with laya english soft dists,
finetunes SystemOne (KL teacher||student + CE + Brier), keeps order-invariance.
"""
import sys, random
sys.path.insert(0, "/content/frontier")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file, save_file
from engine_t4 import SystemOneDecisionEngine
from train_t4 import collate_t4, loss_fn
from synth_data import build_records
from train_t4 import encode_record

DEVICE = "cuda"


def laya_soft(agent, state, labels, ins="Which news topic is this article?"):
    r = agent.predict(state[:2000], {"q": {"type": "choice",
                     "instructions": ins, "criteria": {l: l for l in labels}}})
    return np.array([r["answers"]["q"]["probabilities"][l] for l in labels],
                    dtype=np.float32)


def main(n_ag=400, n_synth=400, epochs=2, lr=2e-5, out="frontier_ckpt_t4_distil"):
    import laya
    from datasets import load_dataset
    device = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained("frontier_ckpt_t4/tokenizer")
    enc = AutoModel.from_pretrained("answerdotai/ModernBERT-base",
                                    attn_implementation="sdpa")
    if device.type == "cuda":
        enc.gradient_checkpointing_enable()
    m = SystemOneDecisionEngine(enc).to(device)
    m.load_state_dict(load_file("frontier_ckpt_t4/model.safetensors"), strict=True)
    m.train()
    print("student loaded", flush=True)

    la = laya.load("convaiinnovations/laya")
    try:
        ag = load_dataset("fancyzhx/ag_news", split="train")
    except Exception:
        ag = load_dataset("sh0416/ag_news", split="train")
    ag_labels = ["World", "Sports", "Business", "Sci/Tech"]
    rng = random.Random(1)
    idx = rng.sample(range(len(ag)), n_ag)
    items = []
    for i in idx:
        row = ag[int(i)]
        try:
            p = laya_soft(la, row["text"], ag_labels)
        except Exception:
            continue
        items.append({"state": row["text"][:2000], "options": list(ag_labels),
                      "soft": [float(v) for v in p], "label": int(row["label"]),
                      "t": "choice"})
    print(f"teacher-labeled AG: {len(items)}/{n_ag}", flush=True)
    recs = build_records(n_per_workflow=max(1, n_synth // 4), seed=7)
    for r in recs[:n_synth]:
        e = encode_record(r)
        try:
            p = laya_soft(la, e["state"], e["options"], ins="Which category applies?")
            e["soft"] = [float(v) for v in p]
        except Exception:
            pass
        items.append(e)
    print(f"total distil items: {len(items)}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for ep in range(epochs):
        tot = 0.0
        order = np.random.RandomState(100 + ep).permutation(len(items))
        import time; t0 = time.time()
        for s in range(0, len(items), 8):
            sel = [items[i] for i in order[s:s + 8]]
            b = collate_t4(sel, tok)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items() if k != "t"}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out_ = m(bd["input_ids"], bd["attention_mask"],
                         bd["option_ids"], bd["option_mask"])
                loss, _ = loss_fn(out_["choice_logits"],
                                  bd["target"].to(out_["choice_logits"].dtype),
                                  bd["marker_mask"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item()
        print(f"epoch {ep+1}/{epochs} loss {tot/max(1,len(items)//8):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    from train_t4 import evaluate
    m.eval()
    print("distil-train acc:", evaluate(m, items[:200], tok, device), flush=True)
    save_file(m.state_dict(), "/tmp/distil.safetensors")
    import os
    os.makedirs(out, exist_ok=True)
    os.system(f"cp /tmp/distil.safetensors {out}/model.safetensors")
    os.system(f"cp -r frontier_ckpt_t4/tokenizer {out}/tokenizer 2>/dev/null; "
              f"cp -r frontier_ckpt_t4/encoder {out}/encoder 2>/dev/null; true")
    enc.config.save_pretrained(f"{out}/encoder")
    tok.save_pretrained(f"{out}/tokenizer")
    import json
    json.dump({"teacher": "convaiinnovations/laya", "n": len(items),
               "epochs": epochs}, open(f"{out}/distil_info.json", "w"), indent=1)
    print(f"saved -> {out} (LOCAL ONLY)", flush=True)


if __name__ == "__main__":
    main()
