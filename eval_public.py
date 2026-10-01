"""Public eval: AG News + Emotion + order-flip, any Frontier ckpt vs recorded Laya.
Usage: python eval_public.py --models frontier_ckpt_v8_ddp [--n_ag 200 --n_em 150]
"""
import argparse, json, sys
sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
sys.path.insert(0, "/teamspace/studios/this_studio/laya")
import numpy as np
from laya.common import ece_score, answer_confidence


def load_agents(tags):
    agents = {}
    for tag in tags:
        if tag == "laya":
            import laya
            agents[tag] = laya.load("convaiinnovations/laya")
        else:
            from agent import FrontierAgent
            d = {"v3": "frontier_ckpt_v3", "v7": "frontier_ckpt_v7",
                 "v8": "frontier_ckpt_v8", "v8_1024": "frontier_ckpt_v8_1024",
                 "v8_ddp": "frontier_ckpt_v8_ddp"}.get(tag, tag)
            import os
            if not os.path.isabs(d):
                d = f"/teamspace/studios/this_studio/frontier/{d}"
            agents[tag] = FrontierAgent(d)
    return agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8_ddp")
    ap.add_argument("--n_ag", type=int, default=200)
    ap.add_argument("--n_em", type=int, default=150)
    a = ap.parse_args()
    from datasets import load_dataset
    try:
        ag = load_dataset("fancyzhx/ag_news", split=f"test[:{a.n_ag}]")
    except Exception:
        ag = load_dataset("sh0416/ag_news", split=f"test[:{a.n_ag}]")
    em = load_dataset("dair-ai/emotion", split=f"test[:{a.n_em}]")
    ag_labels = ["World", "Sports", "Business", "Sci/Tech"]
    em_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    agents = load_agents([x.strip() for x in a.models.split(",") if x.strip()])
    out = {}
    for sname, ds, tcol, lcol, labels in [
            ("ag_news", ag, "text", "label", ag_labels),
            ("emotion", em, "text", "label", em_labels)]:
        for tag, agent in agents.items():
            ok, confs, corr, brier, nll = [], [], [], [], []
            for row in ds:
                y = int(row[lcol])
                try:
                    r = agent.system_one(row[tcol][:2000], {"q": {
                        "type": "choice", "instructions": "Which?",
                        "criteria": {l: l for l in labels}}})
                    p = np.array([r["answers"]["q"]["probabilities"][l] for l in labels])
                except Exception:
                    continue
                pred = int(p.argmax())
                c = pred == y
                ok.append(c); corr.append(c)
                confs.append(answer_confidence(p, len(p)))
                tgt = np.zeros(len(p)); tgt[y] = 1
                brier.append(float(((p - tgt) ** 2).mean()))
                nll.append(float(-np.log(max(p[y], 1e-9))))
            res = {"n": len(ok), "acc": round(float(np.mean(ok)), 4),
                   "ece": round(float(ece_score(np.array(confs), np.array(corr))), 4),
                   "brier": round(float(np.mean(brier)), 4),
                   "nll": round(float(np.mean(nll)), 4)}
            out[f"{tag}/{sname}"] = res
            print(f"{tag}/{sname}: {json.dumps(res)}", flush=True)
    json.dump(out, open("/teamspace/studios/this_studio/frontier/eval_public_v8ddp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
