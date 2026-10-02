"""v9 teacher fill: Laya soft logits for items missing them; alpha by agreement.

alpha: agree 0.5 / disagree 0.1 / existing replay-agree -> 0.5.
Saves teacher_v9.npz {logits[KMAX=8 padded -1e4], alphas, idx} aligned to
data_v9.npz train order. Single GPU, offline; student trains later.
"""
import json, sys
import numpy as np
import torch
from common import content_id, option_hash, kd_alpha, KD_POLICY

KMAX = 8


def main(path="data_v9.npz", out="teacher_v9.npz"):
    z = np.load(path, allow_pickle=True)
    tr = json.loads(str(z["train"]))
    print(f"train items: {len(tr)}", flush=True)
    need = [i for i, r in enumerate(tr) if r.get("teacher") is None]
    print(f"need teacher: {len(need)}", flush=True)
    import laya
    from laya import common as LC
    device = torch.device("cuda")
    agent = laya.load("convaiinnovations/laya")
    tok, model = agent.tok, agent.model.to(device).eval()
    TL = np.full((len(tr), KMAX), -1e4, dtype=np.float32)
    AL = np.zeros(len(tr), dtype=np.float32)
    IDS, OPH, KK = [], [], []
    for i, r in enumerate(tr):
        q = {"t": r["q"]["t"], "ins": r["q"]["ins"], "crit": r["q"]["crit"]}
        IDS.append(content_id(r["state"], q, r["gold"]))
        OPH.append(option_hash(q))
        KK.append(len(q["crit"]) if q["t"] != "noul" else 2)
        if r.get("teacher") is not None:
            t = np.array(r["teacher"], dtype=np.float32)
            TL[i, :len(t)] = t
            AL[i] = kd_alpha(True)  # replay items are agree-by-construction
    B = 32
    with torch.no_grad():
        for s in range(0, len(need), B):
            sel = need[s:s + B]
            enc = []
            for i in sel:
                r = tr[i]
                qq = {"t": r["q"]["t"], "ins": r["q"]["ins"], "crit": r["q"]["crit"] or None}
                if qq["t"] == "choice" and isinstance(qq["crit"], list):
                    qq["crit"] = {c: None for c in qq["crit"]}
                ids, markers = LC.build_sequence(tok, r["state"], qq, 512, 192)
                k = len(markers)
                if k == 0 or k > KMAX:
                    continue
                enc.append((i, {"ids": ids, "markers": markers,
                                "qtype": LC.QTYPES[qq["t"]],
                                "target": [0.0] * k, "label": r["gold"]}))
            if not enc:
                continue
            b = LC.collate_items([[dict(ids=x["ids"], markers=x["markers"], qtype=x["qtype"],
                                         target=x["target"], label=x["label"])] for _, x in enc],
                                 tok.pad_token_id)
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(bd["input_ids"], bd["attention_mask"], bd["marker_pos"],
                                  bd["marker_mask"], bd["qtype"])
            lg = logits.float().cpu().numpy()
            for (i, x), row in zip(enc, lg):
                k = len(x["markers"])
                TL[i, :k] = row[:k]
                agrees = int(np.argmax(row[:k])) == int(tr[i]["gold"])
                AL[i] = kd_alpha(agrees)
                q = {"t": tr[i]["q"]["t"], "ins": tr[i]["q"]["ins"], "crit": tr[i]["q"]["crit"]}
                IDS[i] = content_id(tr[i]["state"], q, tr[i]["gold"])
                OPH[i] = option_hash(q)
                KK[i] = k
            if (s // B + 1) % 20 == 0:
                print(f"  {s+len(sel)}/{len(need)} agree={(AL[:len(tr)]==KD_POLICY['agree']).sum()}", flush=True)
    import time as _t
    meta = {"kd_policy": KD_POLICY, "teacher": "convaiinnovations/laya",
            "teacher_max_len": 512, "teacher_head_len": 192,
            "created_utc": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())}
    np.savez_compressed(out, logits=TL, alphas=AL,
                        ids=np.array(IDS, dtype=object),
                        option_hash=np.array(OPH, dtype=object),
                        num_options=np.array(KK, dtype=np.int32),
                        meta=np.array(json.dumps(meta), dtype=object))
    print(f"saved {out} agree={(AL==0.5).sum()} disagree={(AL==0.1).sum()} none={(AL==0).sum()}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
