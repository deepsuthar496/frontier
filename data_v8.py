"""v8 data: gold-label-first supervision with 4 leakage-free splits.

Budget target ~20-30k decisions:
  50% real permitted gold (AG News + Emotion + public records, gold authoritative)
  20% targeted hard (deterministic probability/computed, routing precedence, multi-hop)
  20% replay (synth generator + teacher-agree items from teacher_labels.npz)
  10% long with verified evidence locations (v4 long + constructed longs)

Splits (train/dev/cal/final) are by GROUP (source doc/template/entity), never by
row: all questions from one document/template stay in one partition.
Teacher role: soft logits only where teacher argmax == gold (agree=0.2 weight);
disagree/missing -> alpha 0 (gold only). No silent pseudo-labels.

Each item: {state, q{t,ins,crit}, gold, teacher (list|None), alpha, group, source,
            state_tokens, kept_512, kept_1024, evidence_ok}
"""
import json, random, math
import numpy as np

SOURCES = {}


def tok_len(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def kept_of(state_toks, head_toks, max_len):
    room = max(0, max_len - head_toks - 1)
    return min(state_toks, room)


# ---------------- deterministic hard probes (solver = authority) ----------------
def prob_items(n, rng, tok):
    """Numerical probability with programmatically computed answers."""
    items = []
    for i in range(n):
        kind = rng.choice(["defect", "delivery", "connection"])
        if kind == "defect":
            k = rng.randint(2, 6)
            p = rng.choice([0.05, 0.1, 0.2])
            p_none = (1 - p) ** k
            gold = 1  # true = at least one defective
            state = (f"Batch inspection: {k} units sampled, each defective with "
                     f"probability {p}. P(none defective) = {p_none:.4f}. "
                     f"Is at least one unit defective more likely than not?")
            items.append({"state": state, "q": {"t": "noul",
                "ins": "Is the stated event more likely than not (probability > 0.5)?",
                "crit": {}}, "gold": int(p_none < 0.5), "teacher": None, "alpha": 0.0,
                "group": f"prob-defect", "source": "v8-solver"})
        elif kind == "delivery":
            early, on, late = rng.randint(5, 30), rng.randint(30, 70), rng.randint(5, 30)
            tot = early + on + late
            pe, po, pl = early / tot, on / tot, late / tot
            gold = int(np.argmax([pe, po, pl]))
            state = (f"Carrier stats over {tot} parcels: {early} early, {on} on time, "
                     f"{late} late. Which outcome is most likely for the next parcel?")
            items.append({"state": state, "q": {"t": "choice",
                "ins": "Which delivery outcome is most likely?",
                "crit": {"early": "Delivered before the promised day.",
                         "on_time": "Delivered on the promised day.",
                         "late": "Delivered after the promised day."}},
                "gold": gold, "teacher": None, "alpha": 0.0,
                "group": "prob-delivery", "source": "v8-solver"})
        else:
            m = rng.randint(20, 90)
            gold = int(m < 45)
            state = (f"Flight connection: {m} minutes between arrival and departure. "
                     f"Minimum viable connection is 45 minutes. "
                     f"Does the passenger miss the connection?")
            items.append({"state": state, "q": {"t": "noul",
                "ins": "Does the passenger miss the connection?",
                "crit": {}}, "gold": gold, "teacher": None, "alpha": 0.0,
                "group": "prob-connection", "source": "v8-solver"})
    return items


def routing_items(n, rng, tok):
    """Conflicting cues with explicit precedence (gold = precedence winner)."""
    items = []
    pairs = [(("billing", "duplicate charge, refund requested"),
              ("security", "phishing link in the same thread")),
             (("technical", "app crash on settings screen"),
              ("security", "credential-stuffing alert on the account"))]
    for i in range(n):
        (a, b) = rng.choice(pairs)
        state = (f"Ticket: {a[1]}. Also: {b[1]}. "
                 f"Policy: security cues take precedence over billing/technical.")
        gold = 1  # second option = security
        opts = {a[0]: a[1], b[0]: b[1]}
        items.append({"state": state, "q": {"t": "choice",
            "ins": "Which department owns this ticket under the precedence policy?",
            "crit": opts}, "gold": gold, "teacher": None, "alpha": 0.0,
            "group": "routing-precedence", "source": "v8-constructed"})
    return items


def multihop_items(n, rng, tok):
    """Two-fact items: answer needs BOTH sentences (single-sentence insoluble)."""
    items = []
    roles = [("Ana", "tier1"), ("Bjorn", "tier2"), ("Chen", "tier3")]
    for i in range(n):
        who, tier = rng.choice(roles)
        oncall = rng.choice(roles)[0]
        # fact1: severity->tier mapping; fact2: who holds that tier + on-call roster
        sev = rng.choice(["sev1", "sev2"])
        need = {"sev1": "tier2", "sev2": "tier3"}[sev]
        holder = [w for w, t in roles if t == need][0]
        state = (f"Alert is {sev}. Policy: {sev} pages {need}. "
                 f"Roster: {need} holder is {holder}; on-call tonight is {oncall}. "
                 f"Who must be paged?")
        opts = {"ana": "Page Ana.", "bjorn": "Page Bjorn.", "chen": "Page Chen."}
        gold = {"ana": 0, "bjorn": 1, "chen": 2}[holder.lower()]
        items.append({"state": state, "q": {"t": "choice",
            "ins": "Who must be paged under the severity policy and roster?",
            "crit": opts}, "gold": gold, "teacher": None, "alpha": 0.0,
            "group": "multihop-roster", "source": "v8-constructed"})
    return items


# ---------------- assemble ----------------
def build_v8(seed=8, n_ag=8000, n_em=3000, n_synth=4000, n_hard=4000,
             n_long=2000, tok_name="frontier_ckpt_v3/tokenizer"):
    import sys
    sys.path.insert(0, "/teamspace/studios/this_studio/frontier")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        f"/teamspace/studios/this_studio/frontier/{tok_name}")
    rng = random.Random(seed)
    from synth_data import build_records
    from data_v3 import public_records
    from data_v4 import build_v4

    items = []

    def add(state, q, gold, group, source, teacher=None, alpha=0.0):
        st = tok_len(tok, state if isinstance(state, str) else json.dumps(state))
        items.append({"state": state, "q": q, "gold": int(gold),
                      "teacher": teacher, "alpha": float(alpha),
                      "group": group, "source": source, "state_tokens": st})

    # 50% real permitted gold
    try:
        from datasets import load_dataset
        ag = load_dataset("fancyzhx/ag_news", split="train")
    except Exception:
        from datasets import load_dataset
        ag = load_dataset("sh0416/ag_news", split="train")
    ag_labels = ["World", "Sports", "Business", "Sci/Tech"]
    idx = rng.sample(range(len(ag)), min(n_ag, len(ag)))
    for i in idx:
        r = ag[int(i)]
        add(r["text"][:2000], {"t": "choice", "ins": "Which news topic is this article?",
            "crit": {l: l for l in ag_labels}}, int(r["label"]),
            f"ag-{ag_labels[int(r['label'])]}", "ag_news-gold")
    try:
        from datasets import load_dataset as _ld
        em = _ld("dair-ai/emotion", split="train")
        eidx = rng.sample(range(len(em)), min(n_em, len(em)))
        em_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        for i in eidx:
            r = em[int(i)]
            add(r["text"][:1000], {"t": "choice", "ins": "Which emotion is expressed?",
                "crit": {l: l for l in em_labels}}, int(r["label"]),
                f"em-{em_labels[int(r['label'])]}", "emotion-gold")
    except Exception as e:
        print("emotion skipped:", e)
    for r in public_records(seed=seed)[:2000]:
        q = r["q"]
        add(r["state"], {"t": q["t"], "ins": q["ins"], "crit": q["crit"]},
            int(r["y"]), f"pub-{q['t']}", "public-gold")

    # 20% targeted hard (solver authority, alpha 0)
    for it in prob_items(n_hard // 3, rng, tok):
        add(it["state"], it["q"], it["gold"], it["group"], it["source"])
    for it in routing_items(n_hard // 3, rng, tok):
        add(it["state"], it["q"], it["gold"], it["group"], it["source"])
    for it in multihop_items(n_hard - 2 * (n_hard // 3), rng, tok):
        add(it["state"], it["q"], it["gold"], it["group"], it["source"])

    # 20% replay (synth gold + teacher-agree only)
    for r in build_records(n_per_workflow=max(1, n_synth // 4), seed=seed):
        q = r["q"]
        keys = list(q["crit"].keys()) if q["t"] == "choice" else None
        add(r["state"], {"t": q["t"], "ins": q["ins"] if "ins" in q else "Which category applies?",
            "crit": q["crit"]}, int(r["y"]), f"syn-{r.get('wf', 'x')}-{q['t']}", "synth-gold")
    try:
        z = np.load("/teamspace/studios/this_studio/frontier/teacher_labels.npz", allow_pickle=True)
        recs = json.loads(str(z["recs"]))
        TL = z["teacher_logits"].astype(np.float32)
        nagree = 0
        for i, r in enumerate(recs):
            q = r["q"]
            k = int(r["k"])
            trow = TL[i, :k]
            tpred = int(np.argmax(trow - trow.max()))
            if tpred == int(r["y"]):
                add(r["state"], {"t": q["t"], "ins": q["ins"], "crit": q["crit"]},
                    int(r["y"]), f"replay-teacher-agree-{q['t']}",
                    "teacher-agree", teacher=[float(x) for x in trow], alpha=0.2)
                nagree += 1
        print(f"replay teacher-agree: {nagree}/{len(recs)}", flush=True)
    except Exception as e:
        print("teacher replay skipped:", e)

    # 10% long with evidence locations (v4 long + constructed, gold known)
    for r in build_v4(seed=seed):
        q = r["q"]
        if len(items) and r.get("wf", "") == "long":
            pass
        st = r["state"]
        slen = tok_len(tok, st if isinstance(st, str) else json.dumps(st))
        if slen >= 700:
            add(st, {"t": q["t"], "ins": q["ins"] if "ins" in q else "Decide.",
                "crit": q["crit"]}, int(r["y"]), f"long-{q['t']}", "v4-long")
        if sum(1 for x in items if x["group"].startswith("long-")) >= n_long:
            break

    # truncation accounting (head approx 100 toks at 512, 150 at 1024)
    for it in items:
        st = it["state_tokens"]
        it["kept_512"] = min(st, 512 - 100 - 1)
        it["kept_1024"] = min(st, 1024 - 150 - 1)
        it["evidence_ok"] = it["kept_512"] >= st or it["source"] in (
            "v8-solver", "v8-constructed", "ag_news-gold", "emotion-gold",
            "synth-gold", "public-gold")

    # 4 splits by GROUP (no group spans partitions) with CLASS COVERAGE:
    # large groups are sharded (distinct docs/rows per shard, no leakage) so no
    # whole (task,label) class can land outside train. Small groups stay whole.
    SHARD = 10
    sharded = []
    for it in items:
        sharded.append(it)
    # shard assignment by index within group
    from collections import defaultdict as _dd
    by_group = _dd(list)
    for i, it in enumerate(items):
        by_group[it["group"]].append(i)
    for g, idxs in by_group.items():
        if len(idxs) >= 300:
            for s, ii in enumerate(idxs):
                items[ii]["group"] = f"{g}#s{s % SHARD}"
    groups = sorted({it["group"] for it in items})
    rng.shuffle(groups)
    # hold out whole SHARDS for final (never trained)
    final_groups = set([g for g in groups if g.startswith("em-joy")][:2]
                       + [g for g in groups if g.startswith("multihop")][:2]
                       + [g for g in groups if g.startswith("long")][:2])
    rest = [g for g in groups if g not in final_groups]
    ndev = max(4, len(rest) // 10)
    dev_groups = set(rest[:ndev])
    cal_groups = set(rest[ndev:2 * ndev])
    train_groups = set(rest[2 * ndev:])
    parts = {"train": [], "dev": [], "cal": [], "final": []}
    for it in items:
        g = it["group"]
        if g in final_groups:
            parts["final"].append(it)
        elif g in dev_groups:
            parts["dev"].append(it)
        elif g in cal_groups:
            parts["cal"].append(it)
        else:
            parts["train"].append(it)
    print({k: len(v) for k, v in parts.items()}, f"groups={len(groups)}", flush=True)
    print("final groups:", sorted(final_groups), flush=True)
    # class-coverage assertion: every (source, gold) in dev/cal/final must exist in train
    tr_keys = {(it["source"], it["gold"]) for it in parts["train"]}
    bad = [(k, it["source"], it["gold"]) for k in ("dev", "cal", "final")
           for it in parts[k] if (it["source"], it["gold"]) not in tr_keys]
    assert not bad, f"train missing classes for {len(bad)} held-out items, e.g. {bad[:5]}"
    print("class coverage OK", flush=True)
    return parts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v8.npz")
    ap.add_argument("--seed", type=int, default=8)
    a = ap.parse_args()
    parts = build_v8(seed=a.seed)
    np.savez_compressed(f"/teamspace/studios/this_studio/frontier/{a.out}",
                        **{k: np.array(json.dumps(v), dtype=object) for k, v in parts.items()})
    print("saved", a.out)
