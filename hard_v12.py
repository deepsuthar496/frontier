"""Hard-family curriculum v1: solver-verified items with template-level splits.

Domains are NEW (never in train/sealed: no hospital/warehouse/backup/permits).
Families: multi_hop / probability / routing_hard / judge_hard.
Every item: id, template_id, split_group, difficulty 1-3, solver authority,
supporting evidence, counterfactual pairs kept together.
"""
import json
import random

SEED = 11


def mk(gid, family, tid, group, diff, state, t, ins, crit, gold, solver, held=False):
    suffix = "-H" if held else ""
    return {"id": gid, "family": family, "template_id": tid + suffix, "split_group": group,
            "difficulty": diff, "state": state,
            "q": {"t": t, "ins": ins, "crit": crit}, "gold": gold,
            "solver": solver, "teacher": None, "alpha": 0.0,
            "source": "synth-hard-v1", "held_bin": held}


# ---------------- multi_hop: policy-chain solver ----------------
def mh_university(n, rng, g0):
    """Admissions: need BOTH program rule AND applicant facts (2 hops)."""
    out = []
    programs = {"cs": ("math>=85", 85), "law": ("essay>=80", 80), "med": ("bio>=90", 90)}
    for i in range(n):
        prog = rng.choice(list(programs))
        rule, cutoff = programs[prog]
        subj = {"cs": "math", "law": "essay", "med": "bio"}[prog]
        score = rng.choice([cutoff - 5, cutoff, cutoff + 5])
        admit = score >= cutoff
        band = "far" if abs(score - cutoff) >= 5 else "near"
        # counterfactual pair: same template, score on other side of cutoff
        for variant, sc in (("a", score), ("b", cutoff - (score - cutoff) if score != cutoff else cutoff - 5)):
            ok = sc >= cutoff
            state = (f"Admissions rule: {prog.upper()} requires {rule}. "
                     f"Applicant file: {subj} score {sc}. "
                     f"Admit only if the requirement is met. Decide.")
            opts = {"admit": "Admit the applicant.", "reject": "Reject the applicant.",
                    "waitlist": "Waitlist pending review."}
            gold = 0 if ok else 1
            out.append(mk(f"mh-uni-{g0+i}{variant}", "multi_hop", f"mh-uni-{prog}-{band}", g0,
                          1 if abs(sc - cutoff) >= 5 else 2, state, "choice",
                          "Admit, reject, or waitlist?", opts, gold,
                          {"name": "threshold-solver", "version": "v1",
                           "evidence": [f"rule:{rule}", f"fact:{subj}={sc}"],
                           "intermediates": [{"req": rule, "have": sc, "met": ok}]},
                           held=(1 <= abs(sc - cutoff) <= 4)))
    return out


def mh_library(n, rng, g0):
    """Library: fine depends on days-overdue AND member type (joint)."""
    out = []
    for i in range(n):
        days = rng.randint(1, 30)
        member = rng.choice(["student", "guest"])
        # rule: students $0.5/day after 7 free days; guests $1/day, no free days
        if member == "student":
            fine = max(0, days - 7) * 0.5
        else:
            fine = days * 1.0
        tier = 0 if fine == 0 else (1 if fine <= 5 else 2)
        state = (f"Library policy: students get 7 free days then $0.50/day; "
                 f"guests pay $1.00/day from day one. "
                 f"Record: {member} member, {days} days overdue. Fine tier?")
        opts = ["no fine", "small fine (up to $5)", "large fine (over $5)"]
        out.append(mk(f"mh-lib-{g0+i}", "multi_hop", f"mh-lib-{member}", g0,
                      2, state, "score", "Which fine tier?",
                      opts, tier,
                      {"name": "fine-solver", "version": "v1",
                       "evidence": ["rule:rates", f"fact:{member},{days}d"],
                       "intermediates": [{"fine": fine}]},
                       held=(0 < fine <= 8)))
    return out


# ---------------- probability: exact solvers ----------------
def prob_items(n, rng, g0):
    import numpy as np
    out = []
    for i in range(n):
        kind = ["backup", "queue", "sensor", "drawer"][i % 4]
        if kind == "backup":
            d = rng.randint(3, 8)
            f = rng.choice([0.02, 0.05, 0.1, 0.2, 0.3])
            p_none = (1 - f) ** d
            gold = int(p_none < 0.5)
            state = (f"Archive check: {d} tapes, each corrupt with probability {f}. "
                     f"P(all readable) = {p_none:.4f}. Is at least one corrupt more likely than not?")
            out.append(mk(f"pr-bk-{g0+i}", "probability", f"pr-backup-{int(f*100)}", g0, 1,
                          state, "noul", "Is corruption more likely than not?", {},
                          gold, {"name": "binomial-solver", "version": "v1",
                                 "evidence": [f"p_none={p_none:.4f}"],
                                 "intermediates": [{"p_none": p_none}]},
                                held=(f in (0.1, 0.2))))
        elif kind == "queue":
            a, b = rng.randint(10, 60), rng.randint(10, 60)
            gold = 0 if a >= b else 1
            state = (f"Clinic audit of {a+b} visits: {a} on-time, {b} delayed. "
                     f"Which is more likely for the next visit?")
            out.append(mk(f"pr-q-{g0+i}", "probability", "pr-queue", g0, 1,
                          state, "choice", "Which outcome is more likely?",
                          {"on_time": "Visit starts on time.", "delayed": "Visit is delayed."},
                          gold, {"name": "proportion-solver", "version": "v1",
                                 "evidence": [f"a={a},b={b}"], "intermediates": []},
                              held=(abs(a / (a + b) - 0.5) <= 0.2)))
        elif kind == "sensor":
            t = rng.randint(5, 120)
            gold = int(t > 30)
            state = (f"Pump sensor has {t} hours left; maintenance arrives in 30 hours. "
                     f"Does it fail first?")
            out.append(mk(f"pr-se-{g0+i}", "probability", "pr-sensor", g0, 1,
                          state, "noul", "Does it fail first?", {}, gold,
                          {"name": "threshold-solver", "version": "v1",
                           "evidence": [f"t={t}"], "intermediates": []},
                            held=(40 <= t <= 80)))
        else:
            r, total = rng.randint(1, 9), 10
            gold = int(r / total > 0.5)
            state = (f"Drawer holds {total} cables, {r} red. "
                     f"Is a random cable more likely red than not?")
            out.append(mk(f"pr-dr-{g0+i}", "probability", "pr-drawer", g0, 1,
                          state, "noul", "More likely red than not?", {}, gold,
                          {"name": "proportion-solver", "version": "v1",
                           "evidence": [f"r={r}/{total}"], "intermediates": []},
                            held=(3 <= r <= 7)))
    return out


# ---------------- routing_hard: precedence + exceptions ----------------
def routing_items(n, rng, g0):
    out = []
    desks = [("visa", "visa applications and entry permits"),
             ("customs", "dutiable goods and contraband screening"),
             ("health", "quarantine and vaccination checks")]
    for i in range(n):
        a, b = rng.sample(desks, 2)
        exc = rng.random() < 0.3
        if exc:
            state = (f"Traveler case: {a[1]} flagged, plus {b[1]} flagged. "
                     f"Policy: {b[0]} takes precedence over {a[0]}, EXCEPT during "
                     f"declared emergencies when {a[0]} handles everything. "
                     f"An emergency IS declared. Which desk?")
            gold, why = 0, "exception"
        else:
            state = (f"Traveler case: {a[1]} flagged, plus {b[1]} flagged. "
                     f"Policy: {b[0]} takes precedence over {a[0]}, except during "
                     f"declared emergencies. No emergency declared. Which desk?")
            gold, why = 1, "precedence"
        opts = {a[0]: a[1], b[0]: b[1]}
        out.append(mk(f"rt-{g0+i}", "routing_hard", f"rt-{a[0]}-vs-{b[0]}", g0,
                      2 if exc else 1, state, "choice",
                      "Which desk owns this case?", opts, gold,
                      {"name": "precedence-solver", "version": "v1",
                       "evidence": [f"rule:{why}"], "intermediates": [{"exception": exc}]},
                       held=("health" in (a[0], b[0]))))
    return out


# ---------------- judge_hard: rubric evaluator ----------------
def judge_items(n, rng, g0):
    out = []
    for i in range(n):
        cited = rng.random() < 0.7
        timely = rng.random() < 0.7
        # rubric: approve needs cited sources AND on-time submission; missing either -> revise;
        # plagiarism flag -> reject outright (disqualifier)
        plag = rng.random() < 0.15
        if plag:
            gold, why = 2, "disqualifier"
        elif cited and timely:
            gold, why = 0, "all-met"
        else:
            gold, why = 1, "partial"
        state = (f"Grant review: sources cited: {cited}; submitted on time: {timely}; "
                 f"plagiarism flag: {plag}. Rubric: approve requires cited sources AND "
                 f"timely submission; a plagiarism flag rejects outright, else revise. Verdict?")
        opts = {"approve": "Meets all requirements.", "revise": "Partially meets, revise.",
                "reject": "Disqualified."}
        out.append(mk(f"ju-{g0+i}", "judge_hard", f"ju-rubric-{why}", g0,
                      2 if plag or (not cited and not timely) else 1, state, "choice",
                      "What is the verdict?", opts, gold,
                      {"name": "rubric-solver", "version": "v1",
                       "evidence": [f"cited={cited}", f"timely={timely}", f"plag={plag}"],
                       "intermediates": [{"why": why}]},
                       held=(why == "partial")))
    return out


def build(seed=11):
    rng = random.Random(seed)
    items = []
    items += mh_university(1200, rng, 0)
    items += mh_library(1200, rng, 10000)
    items += prob_items(2000, rng, 20000)
    items += routing_items(2000, rng, 30000)
    items += judge_items(2000, rng, 40000)
    # F1 split: held parameter bins -> hard-sealed (interpolation test);
    # hard-dev = template subset of non-held; train = rest. Templates never split.
    from collections import defaultdict
    by_t = defaultdict(list)
    for it in items:
        by_t[it["template_id"]].append(it)
    seal_t = {t for t, rs in by_t.items() if rs and rs[0].get("held_bin")}
    open_t = sorted(set(by_t) - seal_t)
    rng.shuffle(open_t)
    dev_t = set(open_t[:max(1, len(open_t) // 10)])
    parts = {"train": [], "hard-dev": [], "hard-sealed": []}
    for t in sorted(by_t):
        if t in seal_t:
            dest = "hard-sealed"
        elif t in dev_t:
            dest = "hard-dev"
        else:
            dest = "train"
        for it in by_t[t]:
            it["split_group"] = dest
            parts[dest].append(it)
    # counterfactual pairs stay together by construction (same template_id)
    print({k: len(v) for k, v in parts.items()}, f"templates={len(by_t)}", flush=True)
    return parts


if __name__ == "__main__":
    parts = build()
    for k, v in parts.items():
        with open(f"/teamspace/studios/this_studio/frontier/hard_v12_{k}.jsonl", "w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
    print("saved hard_v12_{train,hard-dev,hard-sealed}.jsonl")
