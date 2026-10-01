"""JevBench fast smoke (231 public items): laya vs frontier. ~2 min on T4.

Item: {state, question, labels, expected}. Converted to Jev shape; gold mapped per type.
Metrics: accuracy + mean proper score (same 1-normalised-Brier as TDB).
"""
import json, glob, statistics, sys

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")

JB = "/tmp/jevbench/datasets/public"


def to_jev(it):
    q = dict(it["question"])
    t = q["type"]
    if t == "choice" and isinstance(q.get("criteria"), list):
        q["criteria"] = {c: None for c in q["criteria"]}
    return {"q": {"type": t, "instructions": q["instructions"],
                  "criteria": q.get("criteria")}}


def gold_of(it):
    t = it["question"]["type"]
    e = it["expected"]
    if t == "noul":
        return 1 if str(e).lower() in ("yes", "true", "1") else 0
    if t == "score":
        return int(e)
    return str(e)


def proper_score(q, gold, ans):
    if ans is None:
        return 0.0
    if q["type"] == "noul":
        p = ans.get("noul")
        return 1.0 - (float(p) - int(gold)) ** 2 if p is not None else 0.0
    probs = ans.get("probabilities") or {}
    if not probs:
        return 0.0
    g = str(gold)
    keys = set(probs) | {g}
    return 1.0 - sum((float(probs.get(k, 0.0)) - (1.0 if k == g else 0.0)) ** 2 for k in keys) / 2.0


def run(tag, items=None):
    if items is None:
        items = []
        for f in sorted(glob.glob(f"{JB}/*.jsonl")):
            for l in open(f):
                items.append(json.loads(l))
    if tag == "laya":
        import laya
        agent = laya.load("convaiinnovations/laya")
        predict = agent.system_one
    else:
        from agent import FrontierAgent
        d = {"frontier": "frontier_ckpt_v3", "frontier_v6": "frontier_ckpt_v6",
             "frontier_v7": "frontier_ckpt_v7"}[tag]
        agent = FrontierAgent(f"/teamspace/studios/this_studio/frontier/{d}")
        predict = agent.system_one
    acc, scores, by = 0, [], {}
    for it in items:
        try:
            q = to_jev(it)["q"]
            r = predict(it["state"], {"q": q})
            a = r["answers"]["q"]
        except Exception:
            a = None
        g = gold_of(it)
        t = it["question"]["type"]
        if a is not None:
            if t == "noul":
                pred = int(float(a.get("noul", 0)) >= 0.5)
            else:
                pr = a.get("probabilities") or {}
                pred = max(pr, key=lambda k: (pr[k], k)) if pr else None
            ok = int(str(pred) == str(g)) if pred is not None else 0
        else:
            ok = 0
        acc += ok
        s = proper_score({"type": t}, g, a)
        scores.append(s)
        by.setdefault(it.get("family", "?"), []).append(ok)
    n = len(items)
    print(f"[{tag}] n={n} acc={acc/n:.4f} proper={statistics.mean(scores)*100:.2f}", flush=True)
    for k, v in sorted(by.items()):
        print(f"    {k}: {sum(v)/len(v):.3f} (n={len(v)})", flush=True)
    return {"n": n, "acc": round(acc/n, 4), "proper100": round(statistics.mean(scores)*100, 2)}


if __name__ == "__main__":
    import sys as _s
    # fast subset: all easy (48, short) + 12 original + 20 hard, covers
    # choice/noul/score and 18 families without the 3746-token longs
    args = [x for x in _s.argv[1:] if not x.startswith("--")]
    fast = "--fast" in _s.argv
    items = None
    if fast:
        import random as _r
        all_items = []
        for f in sorted(glob.glob(f"{JB}/*.jsonl")):
            for l in open(f):
                all_items.append(json.loads(l))
        easy = [it for it in all_items if it["id"].startswith("easy-")]
        orig = [it for it in all_items if it["id"].startswith("original-")]
        hard = [it for it in all_items if it["id"].startswith("hard-")]
        _rr = _r.Random(0)
        # skip >2000-char states in fast mode (long_policy longs), keep 2 for signal
        longs = [it for it in hard if len(str(it["state"])) > 2000]
        hard_short = [it for it in hard if len(str(it["state"])) <= 2000]
        items = easy + _rr.sample(orig, min(12, len(orig))) \
            + _rr.sample(hard_short, min(18, len(hard_short))) \
            + _rr.sample(longs, min(2, len(longs)))
        _rr.shuffle(items)
        print(f"fast subset: n={len(items)} (easy {len(easy)}, orig 12, hard 20)", flush=True)
    out = {}
    for tag in (args or ["laya", "frontier"]):
        out[tag] = run(tag, items)
    print(json.dumps(out, indent=1))
