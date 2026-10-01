"""Evidence end-to-end: truncated baseline vs chunk-score-select on long items."""
import json, sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")
from agent import FrontierAgent

agent = FrontierAgent("/teamspace/studios/this_studio/frontier/frontier_ckpt_v8_ddp")
def pred_of(t, ans, q):
    if t == "noul":
        return int(float(ans["noul"]) >= 0.5)
    pr = ans.get("probabilities") or {}
    keys = list(q["criteria"].keys()) if t == "choice" else [str(i) for i in range(len(pr))]
    p = [float(pr.get(k, 0.0)) for k in keys]
    return int(max(range(len(p)), key=lambda i: (p[i], -i)))


items = [json.loads(l) for l in open("/tmp/jevbench/datasets/public/hard.jsonl")]
longs = [it for it in items if len(str(it["state"])) > 3000][:6]
print(f"long items: {len(longs)}", flush=True)
for it in longs:
    q = it["question"]
    qdef = {"type": q["type"], "instructions": q["instructions"], "criteria": q.get("criteria")}
    t = q["type"]
    e = it["expected"]
    gold = (1 if str(e).lower() in ("yes", "true", "1") else 0) if t == "noul" else (
        int(e) if t == "score" else list(q["criteria"].keys()).index(str(e)))
    # baseline (truncate)
    agent.cfg["evidence"] = {"enable": False}
    try:
        r0 = agent.system_one(it["state"], {"q": qdef})["answers"]["q"]
        p0 = pred_of(t, r0, q)
    except Exception as ex:
        p0 = f"ERR {str(ex)[:80]}"
    # evidence
    for strat in ["v1", "v2"]:
        agent.cfg["evidence"] = {"enable": True, "top_k": 4, "strategy": strat}
        try:
            r1 = agent.system_one(it["state"], {"q": qdef})["answers"]["q"]
            p1 = pred_of(t, r1, q)
            info = agent._last_evidence.get("q", {})
        except Exception as ex:
            p1, info = f"ERR {str(ex)[:80]}", {}
        print(f"{it['id']} gold={gold} base={p0} ev_{strat}={p1} {info}", flush=True)
