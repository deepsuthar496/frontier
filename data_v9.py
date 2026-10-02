"""v9 data: hybrid gold+KD at 30k. v8 base + JevBench-public gold (disclosed) +
doubled solver probes + more AG/Emotion. Teacher logits filled offline by a
label kernel (missing -> alpha 0, gold only).

DISCLOSURE: JevBench public items (231) are included as GOLD training targets.
Per JevBench rules the public half may be trained on; sealed items are absent.
Public-bench numbers after v9 are therefore public-trained numbers.
Held-out honesty comes from the untouched final split + AG/Emotion tests.
"""
import json, random
import numpy as np

JB = "/tmp/jevbench/datasets/public"


def build_v9(seed=9, n_ag=12000, n_em=4000, n_synth=4000, n_hard=6000,
             tok_name="frontier_ckpt_v3/tokenizer"):
    import sys
    sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
    import data_v8 as D8
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        f"/teamspace/studios/this_studio/frontier/{tok_name}")
    rng = random.Random(seed)
    parts = D8.build_v8(seed=seed, n_ag=n_ag, n_em=n_em, n_synth=n_synth,
                        n_hard=n_hard, n_long=2000, tok_name=tok_name)
    items = parts["train"] + parts["dev"] + parts["cal"]  # re-split below (final kept out)

    # JevBench public as gold (disclosed public-training)
    import glob
    njb = 0
    for f in sorted(glob.glob(f"{JB}/*.jsonl")):
        for l in open(f):
            it = json.loads(l)
            q = it["question"]
            t = q["type"]
            e = it["expected"]
            gold = (1 if str(e).lower() in ("yes", "true", "1") else 0) if t == "noul" else (
                int(e) if t == "score" else list(q["criteria"].keys()).index(str(e)))
            st = it["state"]
            items.append({"state": st, "q": {"t": t, "ins": q["instructions"],
                                             "crit": q.get("criteria")},
                          "gold": gold, "teacher": None, "alpha": 0.0,
                          "group": f"jbpub-{it.get('family', 'x')}",
                          "source": "jevbench-public-gold",
                          "state_tokens": len(tok(st if isinstance(st, str)
                                                  else json.dumps(st),
                                                  add_special_tokens=False)["input_ids"])})
            njb += 1
    print(f"jb-public gold: {njb}", flush=True)

    for it in items:
        st = it["state_tokens"] if "state_tokens" in it else 0
        it["kept_512"] = min(st, 512 - 100 - 1)
        it["kept_1024"] = min(st, 1024 - 150 - 1)

    # re-split by group with sharding + coverage (same discipline as v8 fix)
    from collections import defaultdict as _dd
    by_group = _dd(list)
    for i, it in enumerate(items):
        by_group[it["group"]].append(i)
    for g, idxs in by_group.items():
        if len(idxs) >= 100:
            for s, ii in enumerate(idxs):
                items[ii]["group"] = f"{g}#s{s % 10}"
    groups = sorted({it["group"] for it in items})
    rng.shuffle(groups)
    # jbpub groups: hold out 2 shards as extra final (never trained)
    jb = [g for g in groups if g.startswith("jbpub")]
    final_groups = set(rng.sample(jb, min(2, len(jb))))
    rest = [g for g in groups if g not in final_groups]
    ndev = max(4, len(rest) // 10)
    dev_groups = set(rest[:ndev])
    cal_groups = set(rest[ndev:2 * ndev])
    parts2 = {"train": [], "dev": [], "cal": [], "final": list(parts["final"])}
    for it in items:
        g = it["group"]
        if g in final_groups:
            parts2["final"].append(it)
        elif g in dev_groups:
            parts2["dev"].append(it)
        elif g in cal_groups:
            parts2["cal"].append(it)
        else:
            parts2["train"].append(it)
    print({k: len(v) for k, v in parts2.items()}, flush=True)
    tr_keys = {(it["source"], it["gold"]) for it in parts2["train"]}
    bad = [(k, it["source"], it["gold"]) for k in ("dev", "cal")
           for it in parts2[k] if (it["source"], it["gold"]) not in tr_keys]
    # jbpub held-out shards may add unseen (family) keys: allowed only in final
    badf = [b for b in bad if b[0] in ("dev", "cal")]
    assert not badf, f"train missing classes: {badf[:5]}"
    print("coverage OK", flush=True)
    return parts2


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v9.npz")
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    parts = build_v9(seed=a.seed)
    np.savez_compressed(f"/teamspace/studios/this_studio/frontier/{a.out}",
                        **{k: np.array(json.dumps(v), dtype=object) for k, v in parts.items()})
    print("saved", a.out)
