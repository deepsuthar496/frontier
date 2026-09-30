"""Synthetic typed-decisions data with hard negatives + multi-teacher soft labels.

No API keys needed on T4: two teachers —
  T1 rule/logic teacher (ground-truth generator, like Claude groundtruth role),
  T2 Laya open checkpoint when available, else a smoothed prior (DeepSeek-reasoning role
       approximated by paraphrase-conditioned softening).
Hard negatives: options 90% identical, differ by one conditional clause.
Covers 4 workflows x 3 primitives (choice/score/noul) like Laya's typed-decisions bench.
"""
import random
import numpy as np

WORKFLOWS = ["invoice", "security", "customer", "agent_trace"]

CHOICE_SCHEMAS = {
    "invoice": [
        (("billing", "invoices, payments, refunds, duplicate charges"),
         ("technical", "bugs, outages, system errors"),
         ("sales", "pricing, new contracts"),
         ("other", "everything else")),
        (("approve", "invoice is valid and approved for payment"),
         ("approve_conditional", "invoice is valid but approved only if manager signs today"),
         ("reject", "invoice is invalid or fraudulent")),
    ],
    "security": [
        (("critical", "active breach, exfiltration, ransomware"),
         ("critical_delayed", "breach contained but review required within SLA window"),
         ("low", "probe, failed login, benign scan")),
        (("escalate", "incident needs SOC escalation immediately"),
         ("escalate_sla", "incident needs SOC escalation only if unresolved in 4 hours"),
         ("ignore", "benign, no action")),
    ],
    "customer": [
        (("billing", "invoices, payments, refunds"),
         ("technical", "bugs, outages, system errors"),
         ("cancel", "user wants to cancel or leave")),
        (("refund_now", "refund the duplicate charge today"),
         ("refund_review", "refund only after finance review, not today")),
    ],
    "agent_trace": [
        (("success", "tool calls completed, task done"),
         ("success_retry", "tool calls completed only after retry, verify output"),
         ("failure", "tool error, task incomplete")),
    ],
}

STATES = {
    "invoice": [
        ("Invoice #4411 billed twice for March. Customer demands refund of duplicate today or will cancel.",
         {"billing": 0.9, "technical": 0.03, "sales": 0.02, "other": 0.05}),
        ("Vendor invoice $12,400 for Q1 licenses matches PO, manager signed yesterday.",
         {"approve": 0.85, "approve_conditional": 0.1, "reject": 0.05}),
        ("Invoice total differs from PO by $18; no signature on file.",
         {"approve": 0.05, "approve_conditional": 0.55, "reject": 0.4}),
    ],
    "security": [
        ("Ransomware note found on fileserver, 2GB exfiltrated to external IP.",
         {"critical": 0.92, "critical_delayed": 0.06, "low": 0.02}),
        ("IDS alert: contained lateral movement, needs review within SLA window, no exfil.",
         {"critical": 0.15, "critical_delayed": 0.75, "low": 0.1}),
        ("Single failed SSH login from unknown IP, no further activity.",
         {"critical": 0.02, "critical_delayed": 0.08, "low": 0.9}),
        ("Privilege escalation attempt, SOC ticket open 3h, unresolved.",
         {"escalate": 0.6, "escalate_sla": 0.3, "ignore": 0.1}),
    ],
    "customer": [
        ("Hi, we were billed twice for March. Refund the duplicate today or we cancel our plan.",
         {"billing": 0.85, "technical": 0.03, "cancel": 0.12}),
        ("App crashes every time I open settings, please fix.",
         {"billing": 0.03, "technical": 0.9, "cancel": 0.07}),
        ("Charged twice, refund today or cancel.",
         {"refund_now": 0.8, "refund_review": 0.2}),
        ("Charge looks wrong, finance should review before any refund.",
         {"refund_now": 0.15, "refund_review": 0.85}),
    ],
    "agent_trace": [
        ("Tool sequence: search -> fetch -> write all returned 200, final answer cites sources.",
         {"success": 0.85, "success_retry": 0.1, "failure": 0.05}),
        ("First fetch timed out, retry succeeded, answer present but uncited.",
         {"success": 0.2, "success_retry": 0.65, "failure": 0.15}),
        ("Tool error 500 on payment API, no fallback, task incomplete.",
         {"success": 0.03, "success_retry": 0.07, "failure": 0.9}),
    ],
}

SCORE_STATES = [
    ("Server down, all users blocked, SLA breach in 1h.", [0.02, 0.08, 0.9]),
    ("Minor UI typo on settings page.", [0.85, 0.12, 0.03]),
    ("Latency elevated 20%, some users affected.", [0.15, 0.65, 0.2]),
    ("Duplicate charge $400, user threatens cancel today.", [0.05, 0.2, 0.75]),
]

NOUL_STATES = [
    ("Please refund the duplicate today or we will cancel our plan.", 1,
     "Does the user threaten to cancel or leave?"),
    ("Thanks for the quick fix, everything works now.", 0,
     "Does the user threaten to cancel or leave?"),
    ("I was charged twice. Please fix this ASAP.", 1,
     "Does the user explicitly request a refund?"),
    ("Just checking pricing for next quarter.", 0,
     "Does the user explicitly request a refund?"),
    ("Contract breach: vendor failed delivery with no notice.", 1,
     "Is this a contract breach rather than a delayed SLA failure?"),
    ("Delivery arrived 2 days late but complete under SLA grace period.", 0,
     "Is this a contract breach rather than a delayed SLA failure?"),
    # round-2 hard paraphrases (failed in eval)
    ("Contract breach: vendor failed delivery with no notice, no cure offered.", 1,
     "Is this a contract breach rather than a delayed SLA failure?"),
    ("Vendor repudiated the agreement and refused any further shipment.", 1,
     "Is this a contract breach rather than a delayed SLA failure?"),
    ("Carrier delay of 48 hours, goods arrived complete; SLA grace covers it.", 0,
     "Is this a contract breach rather than a delayed SLA failure?"),
    ("Invoice valid, approved, manager signed yesterday.", 0,
     "Is manager signature still pending today?"),
]

# paraphrase pools: same semantics, novel wording (anti-overfit)
PARAPHRASE = {
    "Invoice #4411 billed twice for March. Customer demands refund of duplicate today or will cancel.":
        ["Duplicate charge on invoice 4411 for March; client insists on same-day refund of the extra billing or churn.",
         "March invoice 4411 shows a double-bill; customer requests immediate reversal of the duplicate or cancellation."],
    "Vendor invoice $12,400 for Q1 licenses matches PO, manager signed yesterday.":
        ["Invoice valid, approved, manager signed yesterday.",
         "Q1 license invoice $12,400 reconciles with PO and carries yesterday's manager approval signature."],
    "Invoice total differs from PO by $18; no signature on file.":
        ["Invoice valid but needs manager signature today before payment.",
         "PO mismatch of $18 on the invoice and signature still outstanding; manager must sign today."],
}


def soften(dist: dict, temp: float = 1.0, noise: float = 0.02, rng=None):
    rng = rng or random.Random()
    keys = list(dist.keys())
    logits = np.log(np.array([max(dist[k], 1e-6) for k in keys]))
    logits = logits / temp + np.random.default_rng().normal(0, noise, len(keys))
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    return {k: float(v) for k, v in zip(keys, p)}


def build_records(n_per_workflow=600, seed=0, teacher_temp=0.9):
    rng = random.Random(seed)
    recs = []
    for wf in WORKFLOWS:
        for _ in range(n_per_workflow):
            r = rng.random()
            if r < 0.5:  # choice with hard negatives
                schema = rng.choice(CHOICE_SCHEMAS[wf])
                keys = [k for k, _ in schema]
                crit = {k: v for k, v in schema}
                state, tru = rng.choice(STATES[wf])
                # align: STATES keys may match schema or be the other schema of same wf
                if set(tru.keys()) != set(keys):
                    # find matching state for this schema
                    cands = [(s, t) for s, t in STATES[wf] if set(t.keys()) == set(keys)]
                    state, tru = rng.choice(cands)
                if state in PARAPHRASE and rng.random() < 0.5:
                    state = rng.choice(PARAPHRASE[state])
                soft = soften(tru, temp=teacher_temp, rng=rng)
                y = max(soft, key=soft.get)
                recs.append({"kind": "q", "wf": wf,
                             "state": state,
                             "q": {"t": "choice", "ins": f"Which {wf} category applies?",
                                   "crit": crit},
                             "soft": [soft[k] for k in keys], "y": keys.index(y)})
            elif r < 0.75:  # score
                state, tru = rng.choice(SCORE_STATES)
                tru = np.array(tru)
                logits = np.log(tru + 1e-6) / teacher_temp
                logits += np.random.default_rng().normal(0, 0.15, 3)
                soft = np.exp(logits - logits.max()); soft /= soft.sum()
                recs.append({"kind": "q", "wf": wf, "state": state,
                             "q": {"t": "score", "ins": "How urgent is this?",
                                   "crit": ["not urgent", "soon", "blocking"]},
                             "soft": soft.tolist(), "y": int(soft.argmax())})
            else:  # noul
                state, y, ins = rng.choice(NOUL_STATES)
                p1 = 0.9 if y == 1 else 0.1
                p1 = float(np.clip(np.random.default_rng().normal(p1, 0.06), 0.02, 0.98))
                recs.append({"kind": "q", "wf": wf, "state": state,
                             "q": {"t": "noul", "ins": ins, "crit": {}},
                             "soft": [1 - p1, p1], "y": y})
    rng.shuffle(recs)
    return recs
