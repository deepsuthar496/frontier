"""v11 data: v9 base + hard curriculum. Template sharding preserved from the
hard generator (template_id wholly in one split). Hard rows carry alpha=0
until a label kernel fills teacher logits; variants set KD policy at train.
"""
import json
import numpy as np


def to_item(r):
    q = r["q"] if "q" in r else {"t": "choice", "ins": "", "crit": {}}
    if "gold_answer" in r:  # spec schema
        t = r["question"]["type"]
        q = {"t": t, "ins": r["question"]["instructions"], "crit": r["question"].get("crit")}
        gold = r["gold_canonical_idx"]
    else:
        gold = r["gold"]
    return {"state": r["state"], "q": q, "gold": int(gold),
            "teacher": r.get("teacher"), "alpha": float(r.get("alpha", 0.0)),
            "group": f"hard-{r.get('template_id', r.get('group', 'x'))}",
            "source": "synth-hard-v1",
            "difficulty": int(r.get("difficulty", 1)),
            "template_id": r.get("template_id", ""),
            "family": r.get("family", ""),
            "state_tokens": int(r.get("state_tokens", 0))}


def build():
    z = np.load("/teamspace/studios/this_studio/frontier/data_v9.npz", allow_pickle=True)
    parts = {k: json.loads(str(z[k])) for k in ["train", "dev", "cal", "final"]}
    n0 = {k: len(v) for k, v in parts.items()}
    hard_tr, hard_dev, hard_se = [], [], []
    for fn, dst in [("hard_v11_train.jsonl", hard_tr),
                    ("hard_v11_hard-dev.jsonl", hard_dev),
                    ("hard_v11_hard-sealed.jsonl", hard_se)]:
        for l in open(f"/teamspace/studios/this_studio/frontier/{fn}"):
            dst.append(to_item(json.loads(l)))
    parts["train"] += hard_tr
    parts["hard-dev"] = hard_dev
    # hard-sealed stays OUT of the npz (separate eval file only)
    print({k: len(v) for k, v in parts.items()}, f"(was {n0})", flush=True)
    print(f"hard-sealed held out: {len(hard_se)} items", flush=True)
    # coverage: every dev/hard-dev (source,gold) present in train
    tr_keys = {(it["source"], it["gold"]) for it in parts["train"]}
    bad = [(k, it["source"], it["gold"]) for k in ("dev", "hard-dev")
           for it in parts[k] if (it["source"], it["gold"]) not in tr_keys]
    assert not bad, f"missing classes: {bad[:5]}"
    print("coverage OK", flush=True)
    np.savez_compressed("/teamspace/studios/this_studio/frontier/data_v11.npz",
                        **{k: np.array(json.dumps(v), dtype=object) for k, v in parts.items()})
    json.dump(hard_se, open("/teamspace/studios/this_studio/frontier/hard_sealed.jsonl", "w"))
    print("saved data_v11.npz + hard_sealed.jsonl")


if __name__ == "__main__":
    build()
