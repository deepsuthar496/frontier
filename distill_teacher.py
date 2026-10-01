"""Teacher-1 distillation labels (spec section 4): Laya-en soft logits over the
full training pool, one fixed option order per item. Student trains with
train=False ordering so KL aligns. Saves teacher_labels.npz (~7.6k items, ~5 min T4).
"""
import json
import os
import random
import sys
import numpy as np
import torch

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")

from synth_data import build_records
from data_v3 import public_records, CHOICE_INS, SCORE_INS
from data_v4 import build_v4

import laya
from laya import common as LC

OUT = "/teamspace/studios/this_studio/frontier/teacher_labels.npz"
KMAX = 8


def materialize(seed=41):
    rng = random.Random(seed)
    syn = build_records(n_per_workflow=700, seed=seed)
    for r in syn:
        if r["q"]["t"] == "choice":
            r["q"] = {"t": "choice", "ins": rng.choice(CHOICE_INS), "crit": r["q"]["crit"]}
        elif r["q"]["t"] == "score":
            r["q"] = {"t": "score", "ins": rng.choice(SCORE_INS), "crit": r["q"]["crit"]}
    pub = public_records(seed=seed)
    new = build_v4(seed=seed)
    recs = syn + pub + new
    rng.shuffle(recs)
    # fix one shuffled option order per item; store materialized record
    items = []
    for r in recs:
        q = r["q"]
        k = len(q["crit"]) if q["t"] == "choice" else (len(q["crit"]) if q["t"] == "score" else 2)
        order = list(range(k))
        if q["t"] != "score":
            rng.shuffle(order)
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            crit = {keys[i]: q["crit"][keys[i]] for i in order}
            soft = [r["soft"][i] for i in order]
            y = order.index(r["y"])
        elif q["t"] == "score":
            crit, soft, y = q["crit"], r["soft"], r["y"]
        else:
            crit, soft, y = {}, r["soft"], r["y"]
        items.append({"state": r["state"],
                      "q": {"t": q["t"], "ins": q["ins"], "crit": crit},
                      "soft": soft, "y": y})
    return items


def main():
    device = torch.device("cuda")
    agent = laya.load("convaiinnovations/laya")
    tok = agent.tok
    model = agent.model.to(device).eval()
    items = materialize()
    print(f"pool: {len(items)} records")
    # encode with LAYA's own codec
    enc = []
    for it in items:
        qq = {"t": it["q"]["t"], "ins": it["q"]["ins"], "crit": it["q"]["crit"] or None}
        if qq["t"] == "choice" and isinstance(qq["crit"], list):
            qq["crit"] = {c: None for c in qq["crit"]}
        ids, markers = LC.build_sequence(tok, it["state"], qq, 512, 192)
        k = len(markers)
        if k != len(it["soft"]) or k > KMAX:
            continue
        enc.append({"ids": ids, "markers": markers, "qtype": LC.QTYPES[qq["t"]],
                    "target": it["soft"], "label": it["y"], "rec": it})
    print(f"encoded: {len(enc)} (dropped {len(items)-len(enc)})")
    TL = np.full((len(enc), KMAX), -1e4, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(enc), 32):
            sel = enc[s:s+32]
            b = LC.collate_items([[dict(ids=x["ids"], markers=x["markers"], qtype=x["qtype"],
                                         target=x["target"], label=x["label"], episode=False,
                                         ep_step=0, ep_len=1)] for x in sel], tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"],
                                  bd["marker_mask"], bd["qtype"])
            lg = logits.float().cpu().numpy()
            for r, x in enumerate(sel):
                TL[s+r, :len(x["markers"])] = lg[r, :len(x["markers"])]
            if (s//32+1) % 10 == 0:
                print(f"  {s+len(sel)}/{len(enc)}", flush=True)
    recs = [{"state": e["rec"]["state"], "q": e["rec"]["q"], "soft": e["rec"]["soft"], "y": e["rec"]["y"],
             "qtype": e["qtype"], "k": len(e["markers"])} for e in enc]
    np.savez_compressed(OUT, teacher_logits=TL,
                        recs=np.array(json.dumps(recs), dtype=object))
    print("saved", OUT, TL.shape)


if __name__ == "__main__":
    main()
