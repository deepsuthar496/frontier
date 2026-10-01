"""Evidence selection for long states: chunk-score-select with the decision
model itself (no new weights), then decide over the selected spans.

Why: multi_hop states run 8-10k chars against a 512/1024 budget (audit kept
ratio 0.27). A wider window alone can't cover 3.7k-token items (1024 stage:
flat at 0.7875). So: split into overlapping chunks, score each chunk with one
batched forward, keep top-k by model confidence, preserve order, decide.

Scoring signal: 1 - normalized entropy of the choice distribution
(common.confidence_from_probs) — the model's own uncertainty, already used
for the confidence field. High-confidence chunks carry decisive evidence;
truncation then drops low-signal filler first.

Cost: one batched forward over short chunks (fast on GPU); only runs when
state_tokens exceed budget, otherwise identity.
"""
import numpy as np
import torch


def chunk_state(tok, state, chunk_tokens=200, overlap=50):
    ids = tok(state, add_special_tokens=False)["input_ids"]
    if len(ids) <= chunk_tokens:
        return [state], [(0, len(ids))]
    chunks, spans = [], []
    step = max(1, chunk_tokens - overlap)
    for s in range(0, len(ids), step):
        e = min(len(ids), s + chunk_tokens)
        chunks.append(tok.decode(ids[s:e]))
        spans.append((s, e))
        if e == len(ids):
            break
    return chunks, spans


@torch.no_grad()
def score_chunks(agent, chunks, qdef):
    """One forward per chunk (batched where possible); returns confidences."""
    from common import QTYPES, build_sequence, collate, confidence_from_probs
    max_len, head_len = agent.cfg["max_len"], agent.cfg["head_max_len"]
    q = agent._to_internal(qdef)
    items = []
    for ch in chunks:
        seq, markers = build_sequence(agent.tok, ch, q, max_len, head_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    b = collate([[it] for it in items], agent.tok.pad_token_id)
    dev = agent.device
    bd = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
    use_amp = dev.type == "cuda"
    with torch.autocast(device_type=dev.type, dtype=agent.dtype, enabled=use_amp):
        out = agent.model(bd["input_ids"], bd["attention_mask"],
                          bd["marker_pos"], bd["marker_mask"], bd["qtype"])
    lg = out["choice_logits"].float().cpu().numpy()
    confs = []
    for r, it in enumerate(items):
        k = len(it["markers"])
        z = lg[r, :k]
        e = np.exp(z - z.max())
        p = e / e.sum()
        confs.append(confidence_from_probs(p, k))
    return confs


def select_evidence(agent, state, qdef, budget_tokens=None, chunk_tokens=200,
                    overlap=50, top_k=4):
    """Return (selected_state, info). Identity when state fits budget."""
    import json as _json
    s = state if isinstance(state, str) else _json.dumps(state, ensure_ascii=False)
    budget = budget_tokens or agent.cfg["max_len"]
    s_ids = agent.tok(s, add_special_tokens=False)["input_ids"]
    if len(s_ids) <= budget:
        return state, {"selected": False, "kept_ratio": 1.0, "n_chunks": 1}
    chunks, spans = chunk_state(agent.tok, s, chunk_tokens, overlap)
    confs = score_chunks(agent, chunks, qdef)
    order = sorted(range(len(chunks)), key=lambda i: -confs[i])[:top_k]
    order = sorted(order)  # preserve document order
    selected = "\n[...]\n".join(chunks[i] for i in order)
    kept = sum(spans[i][1] - spans[i][0] for i in order)
    return selected, {"selected": True, "kept_ratio": round(kept / max(1, len(s_ids)), 3),
                      "n_chunks": len(chunks), "top_conf": round(max(confs), 3),
                      "mean_conf": round(float(np.mean(confs)), 3)}
