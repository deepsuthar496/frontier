"""v8 failure audit: for every item in the zero-scoring families, log mapping,
truncation, and prediction — distinguish capability vs evaluation artifact.

Logs: id, family, type, labels, expected, gold_idx, state_chars, state_tokens,
retained_tokens, truncated, head_tokens, pred, probs, correct, model.
Usage: python audit_v8.py --models v7,laya [--families multi_hop,probability,routing_hard]
"""
import argparse, json, glob, sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")

JB = "/tmp/jevbench/datasets/public"


def load_items(families):
    items = []
    for f in sorted(glob.glob(f"{JB}/*.jsonl")):
        for l in open(f):
            it = json.loads(l)
            if not families or it.get("family") in families:
                items.append(it)
    return items


def gold_idx_of(it):
    t = it["question"]["type"]
    e = it["expected"]
    if t == "noul":
        return 1 if str(e).lower() in ("yes", "true", "1") else 0
    if t == "score":
        return int(e)
    return list(it["question"]["criteria"].keys()).index(str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v7,laya")
    ap.add_argument("--families", default="multi_hop,probability,routing_hard")
    ap.add_argument("--out", default="audit_v8.jsonl")
    a = ap.parse_args()
    fams = [x.strip() for x in a.families.split(",") if x.strip()]
    items = load_items(fams)
    print(f"items: {len(items)} families={fams}", flush=True)

    agents = {}
    for tag in [x.strip() for x in a.models.split(",") if x.strip()]:
        if tag == "laya":
            import laya
            agents[tag] = laya.load("convaiinnovations/laya")
        else:
            from agent import FrontierAgent
            d = {"v3": "frontier_ckpt_v3", "v7": "frontier_ckpt_v7",
                 "frontier": "frontier_ckpt_v3"}[tag]
            agents[tag] = FrontierAgent(f"/teamspace/studios/this_studio/frontier/{d}")
    # tokenizer for token accounting (v7 tokenizer; laya tok is same vocab family)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/teamspace/studios/this_studio/frontier/frontier_ckpt_v7/tokenizer")

    rows = []
    for it in items:
        q = it["question"]
        qdef = {"type": q["type"], "instructions": q["instructions"], "criteria": q.get("criteria")}
        state = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"])
        s_ids = tok(state, add_special_tokens=False)["input_ids"]
        gi = gold_idx_of(it)
        for tag, agent in agents.items():
            try:
                r = agent.system_one(state, {"q": qdef})
                ans = r["answers"]["q"]
                if q["type"] == "noul":
                    p = [1 - float(ans["noul"]), float(ans["noul"])]
                    pred = int(p[1] >= 0.5)
                else:
                    pr = ans.get("probabilities") or {}
                    keys = list(q["criteria"].keys()) if q["type"] == "choice" else [str(i) for i in range(len(it["labels"]))]
                    p = [float(pr.get(k, 0.0)) for k in keys]
                    pred = int(max(range(len(p)), key=lambda i: (p[i], -i)))
                    # note: bench uses min-key tiebreak; argmax-first is equivalent for audit
            except Exception as e:
                ans, p, pred = {"error": str(e)[:200]}, [], -1
            # truncation accounting with v7 geometry (512/192)
            from common import build_sequence, render_options
            qi = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
            try:
                ids, markers = build_sequence(tok, state, qi, 512, 192)
                head_toks = len(ids) - len(s_ids[:max(0, 512 - len(ids))])
                retained = max(0, 512 - (len(ids) - min(len(s_ids), max(0, 512 - len(ids) + 0))))
                room = max(0, 512 - (len(ids) - min(len(s_ids), 512)))
                kept = min(len(s_ids), room)
            except Exception:
                kept = -1
            row = {"id": it["id"], "family": it.get("family"), "type": q["type"],
                   "labels": it["labels"], "expected": it["expected"], "gold_idx": gi,
                   "state_chars": len(state), "state_tokens": len(s_ids),
                   "kept_tokens": kept,
                   "truncated": kept >= 0 and kept < len(s_ids),
                   "model": tag, "pred": pred, "probs": [round(float(x), 4) for x in p],
                   "correct": int(pred == gi)}
            rows.append(row)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # summary: per family x model, accuracy + truncation rate + mean kept/total
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["family"], r["model"])].append(r)
    for (fam, tag), rs in sorted(by.items()):
        acc = sum(r["correct"] for r in rs) / len(rs)
        tr = sum(1 for r in rs if r["truncated"]) / len(rs)
        keepr = sum(r["kept_tokens"] / max(1, r["state_tokens"]) for r in rs) / len(rs)
        print(f"{fam}/{tag}: n={len(rs)} acc={acc:.3f} trunc_rate={tr:.2f} kept_ratio={keepr:.2f}", flush=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
