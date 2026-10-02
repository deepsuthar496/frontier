"""v9 teacher fill: Laya soft logits for items missing them; alpha by agreement.

alpha: agree 0.5 / disagree 0.1 / existing replay-agree -> 0.5.
Saves teacher_v9.npz {logits[KMAX=8 padded -1e4], alphas, idx} aligned to
data_v9.npz train order. Single GPU, offline; student trains later.
"""
import json, sys
import numpy as np
import torch

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
    for i, r in enumerate(tr):
        if r.get("teacher") is not None:
            t = np.array(r["teacher"], dtype=np.float32)
            TL[i, :len(t)] = t
            AL[i] = 0.5  # replay items are agree-by-construction
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
                AL[i] = 0.5 if int(np.argmax(row[:k])) == int(tr[i]["gold"]) else 0.1
            if (s // B + 1) % 20 == 0:
                print(f"  {s+len(sel)}/{len(need)} agree={(AL[:len(tr)]==0.5).sum()}", flush=True)
    np.savez_compressed(out, logits=TL, alphas=AL)
    print(f"saved {out} agree={(AL==0.5).sum()} disagree={(AL==0.1).sum()} none={(AL==0).sum()}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
