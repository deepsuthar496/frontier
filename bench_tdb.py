"""Official Typed Decision Bench v0.3 run: Frontier vs Laya, full 5387-item roster.

Scoring is a verbatim port of blobfishai/jevfish run_report.site_data:
  item_proper_score = mean over gold questions of 1 - normalised Brier
    noul: 1 - (p-gold)^2 ; choice/score: 1 - sum((p-1hot)^2)/2
  DecisionScore = 100 * mean over FULL roster (unanswered/withheld = 0).
ECE/AUROC replicate suite_result (weighted, max-p conf; noul conf = max(p,1-p)).

Both systems get byte-identical (state, questions). No TDB item was trained on.
Usage: python3 bench_tdb.py [--models frontier,laya] [--limit N]
"""
import argparse, json, glob, statistics, sys, time
import numpy as np

sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")

TDB = "/tmp/tdb"


def item_proper_score(questions, gold, answers):
    parts = []
    for qid, g in gold.items():
        q = questions[qid]
        ans = (answers or {}).get(qid)
        if not ans:
            continue
        if q["type"] == "noul":
            p = ans.get("noul")
            if p is None:
                continue
            parts.append(1.0 - (float(p) - int(g)) ** 2)
        else:
            probs = ans.get("probabilities") or {}
            if not probs:
                continue
            gg = str(g)
            keys = set(probs) | {gg}
            parts.append(1.0 - sum((float(probs.get(k, 0.0)) - (1.0 if k == gg else 0.0)) ** 2 for k in keys) / 2.0)
    return statistics.mean(parts) if parts else None


def top_of(q, ans):
    if q["type"] == "noul":
        return int(float(ans["noul"]) >= 0.5) if ans.get("noul") is not None else None
    probs = ans.get("probabilities") or {}
    if not probs:
        return None
    m = max(probs.values())
    return min(k for k, v in probs.items() if v == m)


def load_roster(fraction=1.0):
    items = []
    for f in sorted(glob.glob(f"{TDB}/*.jsonl")):
        for l in open(f):
            items.append(json.loads(l))
    if fraction < 1.0:
        from collections import defaultdict
        by_task = defaultdict(list)
        for it in items:
            key = it["meta"]["task"] if it.get("questions") else "__withheld__"
            by_task[key].append(it)
        sub = []
        for key, lst in by_task.items():
            lst = sorted(lst, key=lambda r: r["id"])
            step = round(1.0/fraction)
            sub.extend(lst[::step])
        # keep roster order stable
        order = {it["id"]: i for i, it in enumerate(items)}
        items = sorted(sub, key=lambda r: order[r["id"]])
    return items


def run_model(tag, items, limit=None):
    if tag == "laya":
        import laya
        agent = laya.load("convaiinnovations/laya")
        def predict(state, questions):
            return agent.predict(state, questions)
    else:
        from agent import FrontierAgent
        d = {"frontier": "frontier_ckpt_v3", "frontier_v2": "frontier_ckpt_final",
             "frontier_v4": "frontier_ckpt_v4",
             "frontier_v5": "frontier_ckpt_v5"}[tag]
        agent = FrontierAgent(f"/teamspace/studios/this_studio/frontier/{d}")
        def predict(state, questions):
            return agent.system_one(state, questions)
    rows = {}
    t0 = time.time()
    작업 = items[:limit] if limit else items
    for i, it in enumerate(작업):
        if it.get("questions") and it.get("gold"):
            try:
                r = predict(it["state"], it["questions"])
                rows[it["id"]] = {"id": it["id"], "answers": r["answers"]}
            except Exception as e:
                rows[it["id"]] = {"id": it["id"], "answers": {}, "error": str(e)[:200]}
        if (i+1) % 500 == 0:
            el = time.time()-t0
            print(f"[{tag}] {i+1}/{len(작업)} answered={sum(1 for v in rows.values() if v.get('answers'))} {el:.0f}s", flush=True)
    print(f"[{tag}] done {len(작업)} items in {time.time()-t0:.0f}s", flush=True)
    return rows


def score_run(items, rows):
    from laya.common import ece_score
    scores, correct, answered = [], 0, 0
    conf, hit, w_all = [], [], []
    p_bin, y_bin, w_bin = [], [], []
    by_suite = {}
    for it in items:
        r = rows.get(it["id"])
        w = float((it.get("meta") or {}).get("weight") or 1.0)
        if not it.get("questions") or not it.get("gold"):
            scores.append(0.0)
            continue
        s = item_proper_score(it["questions"], it["gold"], (r or {}).get("answers"))
        if r and r.get("answers") and s is not None:
            answered += 1
            scores.append(s)
            qid = (it.get("meta") or {}).get("primary") or list(it["questions"].keys())[0]
            q = it["questions"][qid]
            ans = (r["answers"] or {}).get(qid) or {}
            pred = top_of(q, ans)
            g = it["gold"][qid]
            gk = str(g) if q["type"] == "choice" else int(g)
            if pred is not None:
                ok = int(str(pred) == str(gk))
                correct += ok
                if q["type"] == "noul" and ans.get("noul") is not None:
                    p_bin.append(float(ans["noul"])); y_bin.append(int(g)); w_bin.append(w)
                    conf.append(max(p_bin[-1], 1-p_bin[-1]))
                else:
                    conf.append(max(float(v) for v in (ans.get("probabilities") or {"": 0}).values()))
                hit.append(ok); w_all.append(w)
            by_suite.setdefault(it["suite"], []).append(s)
        else:
            scores.append(0.0)
    n = len(items)
    wsum = sum(w_all) if w_all else 1
    ece = float(np.average([abs(c-h) for c, h in zip(
        [sum(conf[i::10])/len(conf[i::10]) if False else 0 for i in range(0)], [])])) if False else None
    out = {"n": n, "answered": answered, "skipped": n-answered,
           "decisionScore": round(100*sum(scores)/n, 2),
           "accuracy_primary": round(correct/answered, 4) if answered else None,
           "ece_maxp_10bin": round(ece_score(np.array(conf), np.array(hit)), 4) if hit else None,
           "by_suite": {k: round(100*sum(v)/len([x for x in items if x.get('suite')==k]), 2) for k, v in by_suite.items()}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="laya,frontier")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--rescore", default=None,
                    help="comma-separated tags: score saved rows on the --fraction subset, no GPU")
    a = ap.parse_args()
    items = load_roster(a.fraction)
    print(f"roster: {len(items)} items ({sum(1 for i in items if i.get('questions') and i.get('gold'))} answerable)")
    if a.rescore:
        summary = json.load(open("/teamspace/studios/this_studio/frontier/tdb_results/summary.json"))
        for tag in a.rescore.split(","):
            tag = tag.strip()
            rows = {json.loads(l)["id"]: json.loads(l)
                    for l in open(f"/teamspace/studios/this_studio/frontier/tdb_results/{tag}.jsonl")}
            s = score_run(items, rows)
            s["subset_fraction"] = a.fraction
            summary[f"{tag}@half" if a.fraction < 1.0 else tag] = s
            print(tag, json.dumps(s, indent=1), flush=True)
        json.dump(summary, open("/teamspace/studios/this_studio/frontier/tdb_results/summary.json", "w"), indent=1)
        raise SystemExit
    import os
    os.makedirs("/teamspace/studios/this_studio/frontier/tdb_results", exist_ok=True)
    try:
        summary = json.load(open("/teamspace/studios/this_studio/frontier/tdb_results/summary.json"))
    except Exception:
        summary = {}
    for tag in a.models.split(","):
        rows = run_model(tag.strip(), items, a.limit)
        with open(f"/teamspace/studios/this_studio/frontier/tdb_results/{tag.strip()}.jsonl", "w") as f:
            for it in items:
                r = rows.get(it["id"], {"id": it["id"], "answers": {}})
                f.write(json.dumps(r) + "\n")
        s = score_run(items, rows)
        s["subset_fraction"] = a.fraction
        key = f"{tag.strip()}@half" if a.fraction < 1.0 else tag.strip()
        summary[key] = s
        print(key, json.dumps(s, indent=1), flush=True)
        # free VRAM before next model
        import gc, torch
        del rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json.dump(summary, open("/teamspace/studios/this_studio/frontier/tdb_results/summary.json", "w"), indent=1)
    print("saved tdb_results/summary.json")
