"""Shared data/codecs: Jev-compatible rendering identical to Laya's rl_common,
plus losses (CE + KL-distill + Brier + RPS) and calibration metrics from guide.md."""
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q) -> list:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else f"{k}: {v}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, q, max_len, head_max_len, option_order=None, truncate_left=False):
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok(f"{q['t']} question: {ins}", add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id] + tok(" " + opts[i].replace(mask_tok, " "),
                                                 add_special_tokens=False)["input_ids"][:48])
    if sum(len(o) for o in opt_ids) > head_max_len - 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
    head_ids = head_ids[:max(8, head_max_len - sum(len(o) for o in opt_ids))]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def collate(records, pad_id: int):
    items = [it for g in records for it in g]
    if not items:
        return None
    L = max(len(it["ids"]) for it in items)
    K = max(len(it["markers"]) for it in items)
    n = len(items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, K), dtype=torch.long)
    mmask = torch.zeros((n, K), dtype=torch.bool)
    tgt = torch.zeros((n, K), dtype=torch.float32)
    labs = torch.full((n,), -1, dtype=torch.long)
    has_label = any("label" in it for it in items)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if "target" in it:
            tgt[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
        if "label" in it:
            labs[i] = it["label"]
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask, "target": tgt,
            "qtype": torch.tensor([it["qtype"] for it in items]),
            "label": labs,
            "n_tokens": int(att.sum())}


# ---------------- losses (guide.md multi-objective) ----------------
def total_loss(student_logits, teacher_logits, target, mask, qtype,
               l_ce=1.0, l_kl=1.0, l_brier=0.5, tau=2.0, w_rps=1.0):
    """CE + tau^2 KL(teacher||student) + Brier + RPS-for-score. All proper."""
    mask_f = mask.float()
    # CE against hard/soft target (masked)
    logp = torch.log_softmax(student_logits, -1)
    ce = -(target * logp * mask_f).sum(-1) / mask_f.sum(-1).clamp_min(1)
    # KL distill (per-option masked: pads carry no teacher mass and must not NaN)
    tp = torch.softmax(teacher_logits / tau, -1).clamp_min(1e-12)
    kl_elem = F.kl_div(torch.log_softmax(student_logits / tau, -1), tp, reduction="none")
    kl = (kl_elem * mask_f).sum(-1) * (tau ** 2)
    # Brier (multi-class, guide.md)
    p = torch.softmax(student_logits, -1) * mask_f
    brier = (((p - target * mask_f) ** 2) * mask_f).sum(-1) / mask_f.sum(-1).clamp_min(1)
    loss = l_ce * ce + l_kl * kl + l_brier * brier
    # RPS for ordinal score questions
    is_score = (qtype == QTYPES["score"])
    if is_score.any():
        k = mask_f.sum(-1).clamp(min=2)
        cdf_q = torch.cumsum(torch.softmax(student_logits, -1) * mask_f, -1)
        cdf_t = torch.cumsum(target * mask_f, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask_f).sum(-1) / (k - 1)
        loss = loss + w_rps * rps * is_score.float()
    return loss.mean(), {"ce": ce.mean().item(), "kl": kl.mean().item(), "brier": brier.mean().item()}


# ---------------- metrics ----------------
def ece_score(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def confidence_from_probs(p, k):
    if k < 2:
        return 1.0
    p = np.asarray(p)[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"


# ---------------- canonical option mapping (E0: one coordinate system) ----------------
# Canonical space: choice -> criteria-key order; score -> ["0"..]; noul -> ["false","true"].
# Model space: marker order after option_order (we always train/eval with identity
# order, so canonical == model; the mapping is explicit so any future shuffle
# cannot silently misalign gold labels, teacher rows, or predictions).
import hashlib as _hl


def canonical_options(q) -> list:
    t = q["t"]
    if t == "choice":
        return list(q["crit"].keys())
    if t == "score":
        return [str(i) for i in range(len(q["crit"]))]
    return ["false", "true"]


def canonical_gold_idx(q, gold) -> int:
    t = q["t"]
    if t == "noul":
        return 1 if str(gold).lower() in ("yes", "true", "1") else 0
    if t == "score":
        return int(gold)
    return canonical_options(q).index(str(gold))


def model_order(q, option_order=None) -> list:
    n = len(canonical_options(q))
    return list(option_order) if option_order is not None else list(range(n))


def gold_model_idx(q, gold, option_order=None) -> int:
    ci = canonical_gold_idx(q, gold)
    return list(model_order(q, option_order)).index(ci)


def decode_pred(model_idx: int, q, option_order=None):
    """Model argmax -> canonical answer VALUE (key string / level int / bool int)."""
    order = model_order(q, option_order)
    ci = order[int(model_idx)]
    t = q["t"]
    if t == "choice":
        return canonical_options(q)[ci]
    if t == "score":
        return int(canonical_options(q)[ci])
    return ci  # noul: 0/1


def content_id(state, q, gold) -> str:
    h = _hl.sha1()
    h.update(serialize_state(state).encode("utf-8", "ignore"))
    h.update(str(q["t"]).encode())
    h.update(str(q.get("ins")).encode())
    h.update(json.dumps(canonical_options(q), ensure_ascii=False).encode())
    h.update(str(gold).encode())
    return h.hexdigest()[:16]


def option_hash(q) -> str:
    return _hl.sha1(json.dumps(canonical_options(q), ensure_ascii=False).encode()).hexdigest()[:16]


# ---------------- KD policy: single source of truth (E0: no drift) ----------------
KD_POLICY = {"agree": 0.5, "disagree": 0.1, "missing": 0.0, "tau": 2.0}


def kd_alpha(teacher_agrees: bool | None) -> float:
    if teacher_agrees is None:
        return KD_POLICY["missing"]
    return KD_POLICY["agree"] if teacher_agrees else KD_POLICY["disagree"]
