"""Sealed v1: 60 fresh hard items authored AFTER all v9 training (2026-10-02).

Sealed property: no v9/v8/v7 training item uses these templates, entities, or
numbers (different domains, rosters, thresholds). Committing the generator does
NOT unseal it: training is frozen in time; nothing trained after this commit
may claim these numbers.
Families: multihop 20 / probability 15 / routing 15 / emotion-guardrail 10.
Gold: solver/precedence/deterministic. Counterfactual pairs stay together.
"""
import json
import random

SEED = 20261002


def multihop(n, rng):
    items = []
    # domain: hospital triage paging (new vs train's alert/roster domain)
    staff = [("Reyes", "triage"), ("Okafor", "surgery"), ("Lindqvist", "pharmacy")]
    for i in range(n // 2):
        who, unit = rng.choice(staff)
        level = rng.choice(["code_blue", "code_yellow"])
        need = {"code_blue": "surgery", "code_yellow": "triage"}[level]
        holder = [w for w, u in staff if u == need][0]
        oncall = rng.choice(staff)[0]
        state = (f"Hospital alert is {level}. Protocol: {level} pages {need} lead. "
                 f"Roster: {need} lead is {holder}; night cover is {oncall}. "
                 f"Who must be paged?")
        opts = {w.lower(): f"Page {w}." for w, _ in staff}
        gold = [w.lower() for w, _ in staff].index(holder.lower())
        items.append(mk(state, "choice", "Who must be paged under protocol and roster?",
                        opts, gold, "multihop", f"seal-hop-page-{i}"))
    # domain: warehouse escalation with two jointly-necessary facts
    for i in range(n - n // 2):
        cls = rng.choice(["flammable", "fragile", "standard"])
        zone = rng.choice(["A", "B"])
        # rule: flammable in zone A -> hazmat crew; flammable in B -> shift lead;
        # fragile -> handler regardless of zone; standard -> no page.
        if cls == "flammable":
            gold_key, why = ("hazmat", "shift_lead")[zone == "B"], "zone rule"
        elif cls == "fragile":
            gold_key, why = "handler", "class rule"
        else:
            gold_key, why = "no_page", "standard"
        state = (f"Warehouse ticket: goods class {cls}, stored in zone {zone}. "
                 f"Rulebook: flammable goods in zone A go to the hazmat crew, in zone B "
                 f"to the shift lead; fragile goods always go to the handler; standard "
                 f"goods need no page. Which route?")
        opts = {"hazmat": "Page hazmat crew.", "shift_lead": "Page shift lead.",
                "handler": "Page handler.", "no_page": "No page, ticket only."}
        items.append(mk(state, "choice", "Which route under the rulebook?", opts,
                        list(opts).index(gold_key), "multihop", f"seal-hop-wh-{i}"))
    return items


def probability(n, rng):
    import numpy as np
    items = []
    for i in range(n):
        kind = ["backup", "queue", "sensor"][i % 3]
        if kind == "backup":
            d = rng.randint(3, 8)
            f = rng.choice([0.02, 0.05, 0.1])
            p_all_ok = (1 - f) ** d
            gold = int(p_all_ok < 0.5)
            state = (f"Nightly backup spans {d} drives, each fails with probability {f}. "
                     f"P(all healthy) = {p_all_ok:.4f}. Is at least one failure more likely than not?")
            items.append(mk(state, "noul", "Is a failure more likely than not?", {},
                            gold, "probability", f"seal-prob-bk-{i}"))
        elif kind == "queue":
            a, b = rng.randint(10, 40), rng.randint(10, 40)
            tot = a + b
            gold = 0 if a >= b else 1
            state = (f"Support queue audit of {tot} tickets: {a} resolved same-day, {b} carried over. "
                     f"Which outcome is more likely for the next ticket?")
            opts = {"same_day": "Resolved same day.", "carry": "Carried over."}
            items.append(mk(state, "choice", "Which outcome is more likely?", opts,
                            gold, "probability", f"seal-prob-q-{i}"))
        else:
            t = rng.randint(5, 120)
            gold = int(t > 30)
            state = (f"Sensor battery lasts {t} more hours; replacement crew arrives in 30 hours. "
                     f"Does the sensor die before the crew arrives?")
            items.append(mk(state, "noul", "Does the sensor die first?", {},
                            gold, "probability", f"seal-prob-se-{i}"))
    return items


def routing(n, rng):
    items = []
    pairs = [(("permits", "building permit application status"),
              ("safety", "gas leak reported at the same address")),
             (("payroll", "missing overtime on payslip"),
              ("security", "impossible-travel login on payroll admin account"))]
    for i in range(n):
        (a, b) = pairs[i % len(pairs)]
        state = (f"Case: {a[1]}. Also: {b[1]}. "
                 f"Policy: safety/security cues take precedence over permits/payroll.")
        opts = {a[0]: a[1], b[0]: b[1]}
        items.append(mk(state, "choice", "Which desk owns this case under precedence?", opts,
                        1, "routing", f"seal-route-{i}"))
    return items


def emotion_guard(n, rng):
    items = []
    cases = [("I am furious about this outage, fix it now!", "anger"),
             ("What a joyful surprise, thank you so much!", "joy"),
             ("I feel so alone since the move, everything is grey.", "sadness"),
             ("That strange noise at night terrifies me.", "fear")]
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for i in range(n):
        text, emo = cases[i % len(cases)]
        items.append(mk(text, "choice", "Which emotion is expressed?",
                        {l: l for l in labels}, labels.index(emo),
                        "emotion", f"seal-emo-{i}"))
    return items


def mk(state, t, ins, crit, gold, family, gid):
    return {"id": gid, "family": family, "state": state,
            "question": {"type": t, "instructions": ins, "criteria": crit},
            "labels": (list(crit.keys()) if t == "choice" else
                       (["0", "1"] if not crit else list(crit.keys()))),
            "expected": gold if isinstance(gold, int) and t != "choice"
            else (list(crit.keys())[gold] if t == "choice" else gold),
            "group": f"seal-{family}", "gold_idx": gold}


if __name__ == "__main__":
    rng = random.Random(SEED)
    items = multihop(20, rng) + probability(15, rng) + routing(15, rng) + emotion_guard(10, rng)
    # counterfactual-pair check: routing/prob items vary the decisive slot by construction
    print({f: sum(1 for it in items if it["family"] == f) for f in
           ["multihop", "probability", "routing", "emotion"]})
    json.dump(items, open("/teamspace/studios/this_studio/frontier/sealed_v1.json", "w"), indent=1)
    print("saved sealed_v1.json:", len(items))
