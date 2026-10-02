"""Full 231-item JevBench public eval: per-model + v7/v8ddp probability ensemble.

Scoring (same as bench_jevbench): accuracy + mean proper score
(1 - normalised Brier; noul 1-(p-g)^2, choice/score 1-sum((p-1hot)^2)/2).
Ensemble averages the two models' probability vectors (identical option
order from shared rendering). No training, no selection.
Records jevbench git rev + item ids for hygiene.
Usage: python bench_full.py --models laya,v7,v8ddp,ens [--jb /tmp/jevbench/datasets/public]
"""
import argparse, glob, json, os, statistics, subprocess, sys

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")


def gold_of(it):
    t = it["question"]["type"]
    e = it["expected"]
    if t == "noul":
        return 1 if str(e).lower() in ("yes", "true", "1") else 0
    if t == "score":
        return int(e)
    return str(e)


def proper_score(t, gold, probs, noul_p=None):
    if t == "noul":
        return 1.0 - (float(noul_p) - int(gold)) ** 2 if noul_p is not None else 0.0
    if not probs:
        return 0.0
    g = str(gold)
    keys = set(probs) | {g}
    return 1.0 - sum((float(probs.get(k, 0.0)) - (1.0 if k == g else 0.0)) ** 2 for k in keys) / 2.0


def answer_of(agent, state, q):
    r = agent.system_one(state, {"q": q})["answers"]["q"]
    t = q["type"]
    if t == "noul":
        return {"noul": float(r["noul"]), "probs": {}}
    pr = r.get("probabilities") or {}
    if t == "choice":
        keys = list(q["criteria"].keys())
    else:
        keys = [str(i) for i in range(len(pr))]
    return {"noul": None, "probs": {k: float(pr.get(k, 0.0)) for k in keys}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="laya,v7,v8ddp,ens")
    ap.add_argument("--jb", default="/tmp/jevbench/datasets/public")
    ap.add_argument("--ckpt_base", default="/teamspace/studios/this_studio/frontier")
    ap.add_argument("--out", default="bench_full.json")
    ap.add_argument("--ens_pair", default="v7,v8ddp",
                    help="two model tags to probability-average as 'ens'")
    a = ap.parse_args()
    items = []
    for f in sorted(glob.glob(f"{a.jb}/*.jsonl")):
        for l in open(f):
            items.append(json.loads(l))
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=a.jb.replace("/datasets/public", ""),
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = "?"
    print(f"items={len(items)} jevbench_rev={rev}", flush=True)

    agents = {}
    tags = [x.strip() for x in a.models.split(",") if x.strip()]
    for tag in tags:
        if tag == "ens":
            continue
        if tag == "laya":
            import laya
            agents[tag] = laya.load("convaiinnovations/laya")
        else:
            from agent import FrontierAgent
            d = {"v3": "frontier_ckpt_v3", "v7": "frontier_ckpt_v7",
                 "v8": "frontier_ckpt_v8", "v8ddp": "frontier_ckpt_v8_ddp",
                 "v9": "frontier_ckpt_v9"}.get(tag, tag)
            if not os.path.isabs(d):
                d = os.path.join(a.ckpt_base, d)
            agents[tag] = FrontierAgent(d)

    results = {t: {"ok": 0, "scores": [], "by": {}} for t in tags}
    for it in items:
        q = {"type": it["question"]["type"], "instructions": it["question"]["instructions"],
             "criteria": it["question"].get("criteria")}
        g, t = gold_of(it), q["type"]
        ans = {}
        for tag, agent in agents.items():
            try:
                ans[tag] = answer_of(agent, it["state"], q)
            except Exception:
                ans[tag] = None
        if "ens" in tags:
            _e1, _e2 = [x.strip() for x in a.ens_pair.split(",")]
            if ans.get(_e1) and ans.get(_e2):
                A, B = ans[_e1], ans[_e2]
                if t == "noul":
                    ans["ens"] = {"noul": (A["noul"] + B["noul"]) / 2, "probs": {}}
                else:
                    keys = set(A["probs"]) | set(B["probs"])
                    ans["ens"] = {"noul": None,
                                  "probs": {k: (A["probs"].get(k, 0.0) + B["probs"].get(k, 0.0)) / 2
                                            for k in keys}}
            else:
                ans["ens"] = None
        for tag in tags:
            an = ans.get(tag)
            if an is None:
                s, okv = 0.0, 0
            elif t == "noul":
                s = proper_score(t, g, {}, an["noul"])
                okv = int(int(float(an["noul"]) >= 0.5) == int(g))
            else:
                s = proper_score(t, g, an["probs"])
                pred = max(an["probs"], key=lambda k: (an["probs"][k], k)) if an["probs"] else None
                okv = int(str(pred) == str(g)) if pred is not None else 0
            results[tag]["ok"] += okv
            results[tag]["scores"].append(s)
            results[tag]["by"].setdefault(it.get("family", "?"), []).append(okv)
    out = {"jevbench_rev": rev, "n": len(items), "ids": [it["id"] for it in items]}
    for tag, r in results.items():
        n = len(items)
        fams = {k: round(sum(v) / len(v), 3) for k, v in sorted(r["by"].items())}
        out[tag] = {"acc": round(r["ok"] / n, 4),
                    "proper100": round(statistics.mean(r["scores"]) * 100, 2),
                    "by_family": fams}
        print(f"[{tag}] acc={r['ok']/n:.4f} proper={statistics.mean(r['scores'])*100:.2f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
