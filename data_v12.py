"""v12 data: v11 base + hard-v12 binned rows. v11 rows kept (volume);
v12 rows carry held_bin discipline (sealed = held bins only)."""
import json
import numpy as np


def to_item(r):
    q = r["q"]
    return {"state": r["state"], "q": q, "gold": int(r["gold"]),
            "teacher": r.get("teacher"), "alpha": float(r.get("alpha", 0.0)),
            "group": f"hard2-{r.get('template_id', 'x')}",
            "source": "synth-hard-v2",
            "difficulty": int(r.get("difficulty", 1)),
            "template_id": r.get("template_id", ""),
            "family": r.get("family", ""),
            "held_bin": bool(r.get("held_bin", False)),
            "state_tokens": 0}


def build():
    z = np.load("/teamspace/studios/this_studio/frontier/data_v11.npz", allow_pickle=True)
    parts = {k: json.loads(str(z[k])) for k in
             ["train", "dev", "cal", "final", "hard-dev"]}
    v12_tr, v12_dev, v12_se = [], [], []
    for fn, dst in [("hard_v12_train.jsonl", v12_tr),
                    ("hard_v12_hard-dev.jsonl", v12_dev),
                    ("hard_v12_hard-sealed.jsonl", v12_se)]:
        for l in open(f"/teamspace/studios/this_studio/frontier/{fn}"):
            dst.append(to_item(json.loads(l)))
    parts["train"] += v12_tr
    parts["hard-dev"] += v12_dev
    # v12 sealed stays OUT (eval file only)
    print({k: len(v) for k, v in parts.items()}, flush=True)
    print(f"v12 hard-sealed held out: {len(v12_se)}", flush=True)
    tr_keys = {(it["source"], it["gold"]) for it in parts["train"]}
    bad = [(k, it["source"], it["gold"]) for k in ("dev", "hard-dev")
           for it in parts[k] if (it["source"], it["gold"]) not in tr_keys]
    assert not bad, f"missing classes: {bad[:5]}"
    print("coverage OK", flush=True)
    np.savez_compressed("/teamspace/studios/this_studio/frontier/data_v12.npz",
                        **{k: np.array(json.dumps(v), dtype=object) for k, v in parts.items()})
    json.dump(v12_se, open("/teamspace/studios/this_studio/frontier/hard_v12_sealed.jsonl", "w"))
    print("saved data_v12.npz + hard_v12_sealed.jsonl")
    # hard label input (v12 train rows only, for the Kev label kernel)
    recs = [{"state": r["state"], "q": r["q"], "gold": r["gold"],
             "teacher": None, "alpha": 0.0,
             "group": r["group"], "source": r["source"]}
            for r in v12_tr]
    np.savez_compressed("/teamspace/studios/this_studio/frontier/hard_v12_label.npz",
                        train=np.array(json.dumps(recs), dtype=object))
    print("saved hard_v12_label.npz:", len(recs))


if __name__ == "__main__":
    build()
